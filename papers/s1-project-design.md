# s1-Lite: 在 Colab T4 上复现 Test-Time Scaling

> 项目设计文档 — 基于 s1: Simple test-time scaling 论文
> 目标平台: Google Colab T4 (15.6 GB VRAM, CUDA 12.8)
> 设计日期: 2026-06-11

---

## 一、项目目标

在 Colab T4 单卡上复现 s1 论文的核心流程：

1. **数据合成**：用教师模型（DeepSeek r1 / Gemini API）为精选推理题生成推理轨迹
2. **SFT微调**：在精选数据上微调小型基座模型（Qwen2.5-7B-Instruct / 14B）
3. **Budget Forcing**：实现论文中的测试时计算控制机制
4. **基准评估**：在 MATH500、AIME 子集上评估 test-time scaling 行为

**成功标准**：
- 基座模型 + SFT 后，在 MATH500 上提升 ≥ 5 个绝对百分点
- Budget Forcing 实现正向 scaling 斜率（更多测试时计算 → 更高准确率）
- 全流程可在 Colab T4 单 session（12h）内完成

---

## 二、平台约束分析

### Colab T4 硬约束

| 资源 | 规格 | 影响 |
|------|------|------|
| GPU 显存 | 15.6 GB | 无法运行32B模型（需16×H100），必须用7B/14B |
| GPU 算力 | ~8 TFLOPS FP16 | 训练时间需严格控制 |
| 系统内存 | ~12 GB | 数据集需全量加载到内存 |
| 磁盘 | ~100 GB | 足够存储模型和数据 |
| 最大Session | 12h（+90min空闲超时） | 需分段训练或一次跑完 |
| CUDA 版本 | 12.8 | Qwen2.5兼容 |

### 模型规模选择

| 模型 | 显存占用 (bf16) | T4可行性 |
|------|----------------|----------|
| Qwen2.5-7B-Instruct | ~14 GB | 边缘可行，需梯度检查点 |
| Qwen2.5-14B-Instruct | ~28 GB | 需要4-bit量化（QLoRA） |
| Qwen2.5-32B-Instruct | ~64 GB | 不可行 |

**推荐方案**：Qwen2.5-7B-Instruct + LoRA/QLoRA — 14B 用 QLoRA 作为 stretch goal。

---

## 三、数据流水线

### 3.1 数据来源

遵循 s1 论文的三原则（质量、难度、多样性），从以下来源精选 ~500 条中文/英文混合推理题：

| 来源 | 预估数量 | 领域 |
|------|----------|------|
| MATH (Hendrycks) | 200 | 竞赛数学 |
| GPQA (子集) | 50 | 博士级科学 |
| AIME 历史题 (1983-2021) | 100 | 竞赛数学 |
| 自建中文推理题 | 50 | 数学/逻辑 |
| OlympicArena (子集) | 100 | 多学科奥林匹克 |

### 3.2 教师模型蒸馏

```
方案A (推荐): DeepSeek r1 API
  - 开源，推理能力强
  - 与 s1.1 论文一致

方案B: Gemini Flash Thinking API (论文原版)
  - 与 s1 原始论文一致

方案C: 低成本替代
  - Qwen2.5-72B-Instruct (HuggingFace 推理端点)
  - 或本地运行 DeepSeek r1-distill 版本
```

### 3.3 数据过滤流水线

```python
# pipeline 伪代码
raw_questions = collect_from_sources()  # 500-1000 题
reasoning_traces = teacher_model.generate(raw_questions)  # 生成推理轨迹

# 质量过滤
filtered = [q for q in reasoning_traces if not has_format_issues(q)]

# 难度过滤 — 用7B模型评估
eval_results = student_model.evaluate(filtered)
hard_questions = [q for q in eval_results if q.accuracy == 0]  # 筛掉简单题

# 多样性采样
domains = classify_domains(hard_questions)  # 使用MSC或自定义分类
selected = diversity_sample(hard_questions, n=500, domains=domains)
```

### 3.4 数据格式

```json
{
  "question": "Find the projection of a onto b = (2,6,3) if a·b = 8.",
  "reasoning_trace": "<|im_start|>think\nFirst, recall the projection formula...\n<|im_start|>answer\n",
  "solution": "proj_b a = (8/49)*(2,6,3) = (16/49, 48/49, 24/49)"
}
```

---

## 四、训练架构

### 4.1 方案：QLoRA + 7B基座

```
基座模型: Qwen2.5-7B-Instruct (bf16)
微调方法: QLoRA (4-bit NF4量化 + LoRA adapters)
LoRA配置:
  - r = 16
  - alpha = 32
  - target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
  - dropout = 0.05
训练参数:
  - epochs: 3-5
  - batch_size: 2 (gradient_accumulation_steps: 8 → effective batch = 16)
  - lr: 2e-4 (QLoRA 通常用更高lr)
  - schedule: cosine with 5% warmup
  - optimizer: paged_adamw_8bit
  - max_seq_length: 4096 (与 s1 论文一致)
  - 只在 reasoning_trace + solution 上计算 loss
分离标记:
  - think_start: "<|im_start|>think"
  - think_end: "<|im_start|>answer"
预计训练时间: 2-4 小时 (T4, ~500 样本)
```

### 4.2 Stretch Goal: 14B + QLoRA

```
基座模型: Qwen2.5-14B-Instruct (4-bit)
显存: ~12-14 GB with QLoRA + gradient checkpointing
训练时间: 4-6 小时
```

---

## 五、Budget Forcing 实现

### 5.1 核心实现

```python
class BudgetForcingController:
    """Budget Forcing 解码控制器"""

    def __init__(self, tokenizer, max_thinking_tokens=None, min_thinking_tokens=None):
        self.tokenizer = tokenizer
        self.max_thinking = max_thinking_tokens
        self.min_thinking = min_thinking_tokens

        # 特殊token
        self.think_end_token = "<|im_start|>answer"
        self.think_start_token = "<|im_start|>think"
        self.wait_string = "Wait"
        self.final_answer = "Final Answer:"

    def intervene(self, generated_text: str, token_count: int) -> str:
        """
        在生成过程中进行干预。
        由 generate() 的回调在每个新token后调用。
        """
        # 强制提前结束
        if self.max_thinking and token_count >= self.max_thinking:
            if self.think_end_token not in generated_text:
                return generated_text + f"\n{self.think_end_token}\n{self.final_answer}"

        # 正在生成 thinking 阶段
        if self.think_end_token not in generated_text:
            return generated_text  # 继续正常生成

        # 已经在 answer 阶段，不干预
        return generated_text

    def extend_thinking(self, generated_text: str) -> str:
        """
        压制 thinking 结束，追加 Wait 延长推理。
        在模型即将产生 end-of-thinking 时调用。
        """
        # 检测模型是否试图结束 thinking
        if self.think_end_token in generated_text[-50:]:
            # 移除 end token，追加 Wait
            text = generated_text.replace(self.think_end_token, "")
            return text + f"\n{self.wait_string}"
        return generated_text
```

### 5.2 集成到 vLLM / HuggingFace

由于 vLLM 不支持解码过程中的自定义干预，有两种方案：

**方案A：HuggingFace generate() + 自定义 LogitsProcessor**

```python
class BudgetForcingLogitsProcessor(LogitsProcessor):
    def __init__(self, tokenizer, max_thinking, think_end_id):
        self.max_thinking = max_thinking
        self.think_end_id = think_end_id
        self.token_count = 0
        self.in_thinking = True

    def __call__(self, input_ids, scores):
        self.token_count += 1
        # 达到上限：强制 model 输出 end-of-thinking token
        if self.in_thinking and self.token_count >= self.max_thinking:
            scores[:] = -float('inf')
            scores[self.think_end_id] = float('inf')
        return scores
```

**方案B：vLLM + 后处理**（仅支持提前结束，不支持延长）

### 5.3 Test-Time Compute 扩展策略

```
配置网格：
  max_thinking_tokens: [512, 1024, 2048, 4096, 8192]
  min_interventions: [0, 1, 2, 4]  (Wait 附加次数)

对每个 (max_tokens, interventions) 组合评估 → 生成 scaling curve
```

---

## 六、评估体系

### 6.1 基准

| 基准 | 题目数 | 指标 | 评估方式 |
|------|--------|------|----------|
| MATH500 | 500 | accuracy | 精确答案匹配 |
| AIME 子集 (2022-2024) | 90 | accuracy | 整数答案匹配 |
| GPQA Diamond (子集) | 100 | accuracy | 选择题匹配 |

### 6.2 s1 论文的三维度量

- **Control**: `I(min_tokens <= actual <= max_tokens)` 的比例
- **Scaling**: accuracy-token 曲线的平均斜率
- **Performance**: 该方法能达到的最高准确率

### 6.3 基线对比

| 基线 | 说明 |
|------|------|
| 基座模型 (无SFT) | Qwen2.5-7B-Instruct 原始推理能力 |
| 基座 + CoT prompting | 标准 Chain-of-Thought 提示 |
| 基座 + BF (无训练) | 在基座上直接应用 Budget Forcing |
| 我们的模型 w/o BF | SFT后不做 test-time 干预 |
| 我们的模型 + BF | **最终目标** |

---

## 七、项目结构

```
projects/s1-lite/
├── README.md                  # 项目概述和使用说明
├── data/
│   ├── collect.py            # 数据收集和清洗
│   ├── distill.py            # 调用教师API生成推理轨迹
│   ├── filter.py             # 难度/质量/多样性过滤
│   └── s1k_lite.jsonl        # 精选数据集
├── train/
│   ├── config.py             # 训练配置
│   ├── dataset.py            # 数据集加载和tokenization
│   ├── train_qlora.py        # QLoRA 训练脚本
│   └── merge_lora.py         # LoRA权重合并
├── inference/
│   ├── budget_forcing.py     # Budget Forcing 核心实现
│   ├── generate.py           # 推理和评估脚本
│   └── scaling_eval.py       # 多预算点 scaling 评估
├── eval/
│   ├── metrics.py            # Control/Scaling/Performance 计算
│   ├── math500_eval.py       # MATH500 评估
│   └── aime_eval.py          # AIME 评估
├── launch.py                 # Colab 一键启动
├── check_progress.py         # 训练监控
└── charts.py                 # 结果可视化
```

---

## 八、实施路线图

### Phase 1: 基础设施 (预计 2-3h)

- [ ] 在 Colab T4 上验证 Qwen2.5-7B-Instruct 可加载 (QLoRA quantized)
- [ ] 测试基础推理生成，测量延迟/显存
- [ ] 搭建 HuggingFace 训练管线（QLoRA + SFT）
- [ ] 验证 500 条数据可正常加载和训练

### Phase 2: 数据准备 (预计 2-3h)

- [ ] 收集 500-1000 条推理题（MATH + AIME + 其他来源）
- [ ] 用教师模型生成推理轨迹（调用 API 或本地推理）
- [ ] 执行质量/难度/多样性过滤
- [ ] 格式化为 `<think>...</think><answer>...</answer>` 格式

### Phase 3: SFT 训练 (预计 3-4h)

- [ ] 实现 loss 只在 reasoning+answer 上计算
- [ ] 训练 QLoRA adapters（3-5 epochs）
- [ ] 保存并合并 LoRA 权重
- [ ] 验证模型可正常加载和推理

### Phase 4: Budget Forcing (预计 2-3h)

- [ ] 实现 LogitsProcessor 版本的 BF
- [ ] 测试提前结束（max_tokens）功能
- [ ] 测试延长推理（suppress + "Wait"）功能
- [ ] 验证正向 scaling: 更多 token → 更高准确率

### Phase 5: 评估与可视化 (预计 2h)

- [ ] 在 MATH500 上评估
- [ ] 多预算点 scaling 曲线绘制
- [ ] 与基座模型对比
- [ ] 撰写实验报告

---

## 九、风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|----------|
| 7B模型推理能力不足 | 中 | 高 | 预先评估基座模型在 MATH500 上的表现；切换到14B+QLoRA |
| 训练在T4上OOM | 中 | 中 | 减小batch、缩短序列、使用gradient checkpointing |
| 蒸馏数据质量差 | 低 | 高 | 人工审核30-50条样本；多种教师模型对比 |
| BF不产生正向scaling | 中 | 高 | 验证基座模型是否已有强推理能力；尝试不同BF字符串 |
| Colab session中断 | 高 | 低 | 频繁保存checkpoint；Drive mount持久化 |

---

## 十、与 s1 论文的差异与简化

| 维度 | s1 论文 | s1-Lite |
|------|---------|---------|
| 基座模型 | Qwen2.5-32B-Instruct | Qwen2.5-7B-Instruct |
| 训练方法 | Full SFT (bf16) | QLoRA (4-bit) |
| 数据量 | 1,000 (59K中精选) | ~500 (更少来源精选) |
| GPU | 16 × H100 | 1 × T4 |
| 训练时间 | 26 min (16 × H100) | 2-4 h (1 × T4) |
| 评估范围 | AIME24, MATH500, GPQA | MATH500 + AIME子集 |
| BF扩展范围 | 1x-32x | 1x-6x |
| 并行扩展 | REBASE + 多数投票 | 仅序列扩展 |

**核心保留**：
- 数据精选的三原则（质量、难度、多样性）
- Budget Forcing 的完整机制
- 三维度量（Control、Scaling、Performance）
- SFT 训练仅针对 reasoning + answer

---

## 十一、预期结果

基于论文结果外推，保守预期：

| 模型 | MATH500 |
|------|---------|
| Qwen2.5-7B-Instruct (基座) | ~60-65% |
| s1-Lite (SFT, w/o BF) | ~70-75% |
| s1-Lite + BF | ~75-80% |

关键验证点：
1. SFT 是否带来了统计学显著提升（≥5个百分点）
2. Budget Forcing 是否产生单调正向的 accuracy-token 曲线
3. 更大的 max_tokens 预算是否持续改善结果（在达到饱和前）

---

*设计完成于 2026-06-11*
