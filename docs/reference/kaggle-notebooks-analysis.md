# Kaggle Notebooks 分析报告：作为 Colab 免费 GPU 训练的补充方案

> 撰写日期：2026-06-11
> 背景：当前使用 Google Colab 免费版进行 GPU 训练，面临会话不稳定（中国代理环境下 WebSocket 断连）、每账号仅 1 GPU、配额不透明等问题。本报告评估 Kaggle Notebooks 作为补充方案的可行性。

---

## 目录

1. [Kaggle 平台概述](#1-kaggle-平台概述)
2. [Kaggle API / CLI 深入分析](#2-kaggle-api--cli-深入分析)
3. [MCP 服务器可行性](#3-mcp-服务器可行性)
4. [Colab vs Kaggle 对比](#4-colab-vs-kaggle-对比)
5. [集成策略](#5-集成策略)
6. [建议与优先级](#6-建议与优先级)

---

## 1. Kaggle 平台概述

### 1.1 免费 GPU 配额

Kaggle Notebooks 提供目前所有免费平台中最慷慨的 GPU 额度：

| 资源 | 限制 | 备注 |
|------|------|------|
| **每周 GPU 时长** | **30 小时** | 每周重置，配额可见且可预测 |
| **单次会话上限** | **12 小时** | 强制终止，需提前保存检查点 |
| **空闲超时** | ~60 分钟 | 非活跃会话会被回收 |
| **加速器类型** | P100 (16GB) 或 **T4 x2 (双卡，~32GB)** | Kaggle 自动分配，不可手动选择 |
| **CPU 内存** | ~16 GB | 优于 Colab 的 ~12 GB |
| **磁盘空间** | `/kaggle/working/` ~20 GB | 会话结束后不持久保存 |

**关键优势**：T4 x2 双卡配置提供约 32GB 总 VRAM，可以运行 Colab 单 T4 无法承载的模型（如 7B 参数的 QLoRA 微调）。

### 1.2 GPU 使用限制详解

- **配额计数器**：Kaggle UI 右上角可见实时 GPU 使用时间（`30:00 / 30:00`），透明可控
- **排队机制**：高峰时段可能需要等待 GPU 分配，但等待时间通常可接受
- **电话验证**：首次使用 GPU / 互联网必须完成电话验证。从中国接收验证短信可能存在困难（Google reCAPTCHA 被屏蔽），常见解决方法：
  - 使用手机端 VPN 再进行验证
  - 使用浏览器扩展（如 Header Editor）重定向 Google 验证服务

### 1.3 目录结构

| 目录 | 权限 | 大小限制 | 持久性 |
|------|------|----------|--------|
| `/kaggle/input/` | **只读** | ~107 GB (私有数据集) / 无限制 (公开) | 由数据集本身决定 |
| `/kaggle/working/` | **读写** | ~20 GB | **会话结束即丢失**，需通过 "Save & Run All" 保存 |
| `/kaggle/temp/` | 读写 | 不定 | 仅当前会话有效 |

**重要**：`/kaggle/working/` 中的文件在会话结束后 **不会自动保留**。必须通过以下方式持久化：
1. **"Save Version" > "Save & Run All"** — 在后台重新执行 notebook 并保存输出文件
2. **输出 → 创建数据集** — 将输出文件转为 Kaggle 数据集，供后续 Notebook 通过 `/kaggle/input/` 访问
3. **外部存储** — 训练结束时手动上传到 Google Drive、HuggingFace Hub 或直接下载

### 1.4 从中国访问的可用性

- **Kaggle.com 本身未被全面封锁**，但 Google reCAPTCHA（注册/登录用）被 GFW 拦截
- **登录后的会话可以正常使用**，但需要已注册的账号
- **开启 Internet 后**，pip 安装可使用国内镜像加速：
  ```python
  !pip install <package> -i https://pypi.tuna.tsinghua.edu.cn/simple/
  ```
- **建议**：一旦完成电话验证和登录，后续使用（CLI API）不再需要验证页面，中国用户可正常使用

---

## 2. Kaggle API / CLI 深入分析

### 2.1 官方工具链

Kaggle 提供两套官方 Python 工具：

| 工具 | 类型 | 安装 | 用途 |
|------|------|------|------|
| **`kaggle` CLI** | 命令行工具 | `pip install kaggle` | 数据上传下载、Notebook 推送、竞赛操作 |
| **`kagglehub`** | Python 库 | `pip install kagglehub` | ML 流水线中的数据/模型下载，Colab 已原生集成 |

### 2.2 `kaggle` CLI 安装与认证

```bash
# 安装
pip install kaggle

# 认证方式一：配置文件
mkdir -p ~/.kaggle
# 从 https://www.kaggle.com/settings 下载 API Token (kaggle.json)
# 文件格式：{"username":"xxx","key":"xxxxxxxx"}
cp ~/Downloads/kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 认证方式二：环境变量
export KAGGLE_USERNAME="your-username"
export KAGGLE_KEY="your-api-key"
```

### 2.3 Notebook / Script 管理命令

这是本报告最核心的部分——这些命令让 Kaggle Notebooks 成为一个可编程的 GPU 训练平台。

| 命令 | 功能 |
|------|------|
| `kaggle kernels init -p <dir>` | 生成 `kernel-metadata.json` 模板 |
| `kaggle kernels push -p <dir>` | **上传并运行** Notebook/Script（核心命令） |
| `kaggle kernels status <owner>/<slug>` | 查看运行状态（running/complete/error） |
| `kaggle kernels output <owner>/<slug>` | 下载输出文件 |
| `kaggle kernels pull <owner>/<slug>` | 下载 Notebook 源码 |
| `kaggle kernels list -s <keyword>` | 搜索公开 Notebook |

### 2.4 `kernel-metadata.json` 完整配置

这是 Kaggle 自动化的关键文件，相当于 Colab 的 `launch.py`：

```json
{
  "id": "your-username/your-kernel-slug",
  "title": "Your Training Script",
  "code_file": "train.py",
  "language": "python",
  "kernel_type": "script",
  "is_private": true,
  "enable_gpu": true,
  "enable_internet": true,
  "dataset_sources": [
    "your-username/your-training-data"
  ],
  "kernel_sources": [],
  "competition_sources": [],
  "model_sources": []
}
```

**关键参数说明**：

- `kernel_type`: `"notebook"`（.ipynb）或 `"script"`（.py）—— **支持纯 Python 脚本**！
- `enable_gpu`: `true` 启用 GPU 加速
- `enable_internet`: `true` 允许访问外部网络（pip 安装等）
- `is_private`: `true` 私有 Notebook，其他人不可见
- `dataset_sources`: 需要挂载的数据集列表
- `id`: 首次推送后会自动更新为实际 ID

### 2.5 Notebook vs Script 模式

Kaggle 支持两种内核类型：

| 类型 | 文件格式 | 适用场景 |
|------|----------|----------|
| **Notebook** | `.ipynb` | 探索性数据分析、可视化、教程分享 |
| **Script** | `.py` | **批量训练、自动化流水线、竞赛提交** |

**Script 模式的优势**：

- 可以直接使用你现有的 `train.py`，无需转换为 notebook 格式
- 执行方式是从上到下顺序执行，行为可预测
- 更易于版本控制和 CI/CD 集成

### 2.6 完整推送工作流

```bash
# 1. 创建项目目录
mkdir -p my-kaggle-experiment
cd my-kaggle-experiment

# 2. 放入训练脚本
cp /path/to/train.py .

# 3. 生成元数据模板
kaggle kernels init -p .

# 4. 编辑 kernel-metadata.json（如上所示）

# 5. 推送并运行
kaggle kernels push -p .

# 6. 查看状态
kaggle kernels status your-username/my-training-script

# 7. 运行完成后下载输出
kaggle kernels output your-username/my-training-script -p ./output
```

**实际输出示例**：

```
$ kaggle kernels push -p .
Kernel push completed. Kernel slug: your-username/my-training-script
Kernel status: running
Kernel output will be available at:
https://www.kaggle.com/code/your-username/my-training-script
```

### 2.7 超参数扫描自动化

Kaggle CLI 的推送模式天然支持实验迭代：

```bash
#!/bin/bash
# 批量超参数扫描
for lr in 0.001 0.0005 0.0001; do
  for bs in 32 64; do
    sed -i "s/learning_rate = .*/learning_rate = $lr/" train.py
    sed -i "s/batch_size = .*/batch_size = $bs/" train.py
    git add . && git commit -m "exp: lr=$lr bs=$bs"
    kaggle kernels push -p .
    sleep 300  # 等待前一个实验启动
  done
done
```

### 2.8 数据集管理

```bash
# 创建/上传数据集
kaggle datasets create -p ./dataset-dir --dir-mode zip

# 下载数据集
kaggle datasets download your-username/dataset-name

# 列出版本
kaggle datasets status your-username/dataset-name

# 更新数据集版本
kaggle datasets version -p ./dataset-dir -m "update message"
```

### 2.9 `kagglehub`——新一代 Python 库

`kagglehub` 是 Kaggle 官方推出的轻量 Python 库，专注于 ML 流水线集成：

```python
import kagglehub

# 下载数据集（返回本地路径）
path = kagglehub.dataset_download("owner/dataset-name")

# 下载模型
model_path = kagglehub.model_download("owner/model-name")
```

**重要更新**：Google Colab 已在 2025 年底原生集成 KaggleHub，通过 Colab Data Explorer 可直接搜索挂载 Kaggle 数据集。

### 2.10 API 限制与注意事项

| 操作 | 是否支持 | 备注 |
|------|----------|------|
| 推送 Notebook/Script 运行 | **支持** | `kaggle kernels push` |
| 查看运行状态 | **支持** | `kaggle kernels status` |
| 下载运行输出 | **支持** | `kaggle kernels output` |
| 终止运行中的 Notebook | **不支持直接 API 调用** | 需通过浏览器 UI 操作 |
| 自动提交竞赛结果 | **不支持** | 2026 年起 API 返回 403 |
| 创建 Utility Script | **有限支持** | 需先从 UI 创建再拉取修改 |
| 指定 GPU 类型 (P100 vs T4) | **不支持** | Kaggle 自动分配 |

---

## 3. MCP 服务器可行性

### 3.1 现有 MCP 服务器

截至 2026 年 6 月，GitHub 上已有多个 Kaggle MCP 服务器实现：

| 项目 | 工具数 | 特点 | 语言 |
|------|--------|------|------|
| **[Seif-Sameh/Kaggle-mcp](https://github.com/Seif-Sameh/Kaggle-mcp)** | **39 个工具** | 最全面：竞赛(8) + 数据集(10) + 内核(7) + 模型(14) | Python |
| **[Galaxy-Dawn/kaggle-mcp](https://github.com/Galaxy-Dawn/kaggle-mcp) | 21 个工具 | 包含讨论区管理，支持 KGAT 新认证 | Python |
| **[arrismo/kaggle-mcp](https://github.com/arrismo/kaggle-mcp)** | 2 个工具 | 轻量级搜索/下载数据集 | Python |
| **[Dishant27/kaggle-MCP](https://github.com/Dishant27/kaggle-MCP)** | ~6 个工具 | 聚焦竞赛提交，Node.js 实现 | TypeScript |
| **[dexhunter/kaggle-mcp](https://github.com/dexhunter/kaggle-mcp)** | 少量 | 早期版本 v0.1.0 | Python |

### 3.2 `Seif-Sameh/Kaggle-mcp` 功能详述

这是目前功能最全面的 Kaggle MCP 服务器，提供 39 个工具，几乎覆盖 Kaggle API 全部功能：

**数据类**（10 个工具）：
- `download_dataset_file`、`download_dataset_files`、`dataset_list`、`dataset_view`、`dataset_download`、`dataset_upload`、`dataset_create`、`dataset_initialize`、`dataset_new_version`、`dataset_update_dataset`

**内核类**（7 个工具）：
- `kernels_list`、`kernels_push`、`kernels_pull`、`kernels_initialize`、`kernels_status`、`kernels_output`、`kernels_kernels_list`

**竞赛类**（8 个工具）：
- `competitions_list`、`competitions_list_leaderboard`、`competitions_download`、`competitions_submissions`、`competitions_submit`、`competitions_competitions_list_leaderboard`、`competitions_competitions_submissions`、`competitions_create_submission`

### 3.3 MCP 服务器的价值

对于 colab-cli 项目，Kaggle MCP 服务器的核心价值在于：

1. **为 Claude Agent 提供 Kaggle 操作接口**：通过 MCP 协议，Claude 可以直接调用 Kaggle 的 API，无需手动拼接命令行
2. **自动化训练流水线**：上传数据集 -> 推送训练脚本 -> 监控状态 -> 下载结果，全流程自动化
3. **多账号管理**：结合多个 Kaggle 账号的 GPU 配额，最大化免费 GPU 使用效率

### 3.4 是否需要自建 MCP 服务器

| 需求 | 现有方案 | 结论 |
|------|----------|------|
| 基础数据集操作 | 所有 MCP 都支持 | 无需自建 |
| Notebook/Script 推送 | Seif-Sameh 支持 | 无需自建 |
| 运行状态监控 | Seif-Sameh 支持 | 无需自建 |
| **多账号轮询 GPU 配额** | **无现有支持** | **可能需要自建** |
| **Colab ↔ Kaggle 工作流桥接** | **无现有支持** | **可能需要自建** |
| **从中国代理优化** | **无现有支持** | **可能需要自建** |

**建议**：优先使用现有的 `Seif-Sameh/Kaggle-mcp`，在此基础上扩展多账号管理功能。不需要从零开始构建。

---

## 4. Colab vs Kaggle 对比

### 4.1 功能矩阵

| 维度 | Google Colab Free | Kaggle Notebooks Free |
|------|------------------|----------------------|
| **GPU 类型** | T4 (16GB) | P100 (16GB) 或 T4 x2 (~32GB) |
| **每周 GPU 配额** | 不透明、动态 | **30 小时，透明可查** |
| **单次会话时长** | ~12 小时 | ~12 小时（GPU） |
| **空闲超时** | ~90 分钟 | ~60 分钟 |
| **CPU 内存** | ~12 GB | ~16 GB |
| **TPU 支持** | **有 (v2/v3)** | 无 |
| **Google Drive 挂载** | **原生支持** | 不支持 |
| **自定义 CUDA 版本** | 较灵活 | 固定（CUDA 11.8+） |

### 4.2 自动化能力对比

| 维度 | Colab CLI (colab-cli) | Kaggle CLI |
|------|----------------------|------------|
| **创建新会话** | `colab new` | 无需（push 自动创建） |
| **上传文件** | `colab upload` | 通过数据集 API |
| **执行脚本** | `colab exec -f script.py` | `kaggle kernels push`（包装为 .py） |
| **后台执行** | `nohup` + `start_new_session` | **原生支持**（push 即后台运行） |
| **查看状态** | `colab list`, `check_progress` | `kaggle kernels status` |
| **下载输出** | `colab download` | `kaggle kernels output` |
| **停止会话** | `colab stop` | **无 API 支持**（需 UI 操作） |
| **保持活跃** | 需 watchdog 心跳 | **不需要，push 运行状态不受浏览器影响** |

### 4.3 中国用户可用性对比

| 维度 | Colab | Kaggle |
|------|-------|--------|
| **WebSocket 稳定性** | **差**——从中国通过代理经常断连 | **不依赖 WebSocket**（REST API 模型） |
| **页面访问** | 需要代理 | 登录后正常，注册需要处理 reCAPTCHA |
| **pip 镜像** | 需要设置 | 支持国内镜像 |
| **文件上传** | `colab upload` 通过 CLI，相对稳定 | 通过数据集 API，也是 REST |
| **主要坑点** | 代理 WebSocket 断连导致训练中断 | 电话验证需处理 GFW |

### 4.4 关键差异分析

**Kaggle 的显著优势**：

1. **后台执行模型（Push 模式）**：`kaggle kernels push` 是一次性的 REST API 调用，不依赖长连接。推送后即可关闭本地终端，Notebook 在 Kaggle 服务器上继续运行。相比之下，Colab 的 `colab exec` 依赖 SSH 或 WebSocket 长连接，从中国使用代理时极不稳定。

2. **配额透明**：30 小时/周的配额清晰可见，可以提前规划训练任务。Colab 的 GPU 配额是动态的，有时连续可用，有时刚启动就被拒绝。

3. **双 GPU 潜力**：T4 x2 配置让 7B 参数模型的 QLoRA 微调成为可能，而 Colab 的单 T4 在这方面受限。

**Colab 的独特优势**：

1. **TPU 支持**：对于 TPU 优化的工作负载（如 JAX 生态），Colab 是唯一选择。
2. **Google Drive 集成**：数据集和检查点可以直接读写 Drive，不需要手动管理数据集版本。
3. **更灵活的 CUDA 版本**：可以安装自定义版本的 CUDA/PyTorch，Kaggle 的环境相对固定。

---

## 5. 集成策略

### 5.1 核心思路

Kaggle 不应完全替代 Colab，而是作为 **Colab 不稳定时的备用 GPU 资源**。两套工作流共享同一份训练代码。

### 5.2 共享代码结构

```
projects/my-project/
├── train.py              # 共享训练脚本（两平台共用）
├── kernel-metadata.json  # Kaggle 推送配置
├── launch.py             # Colab 启动脚本
├── check_progress.py     # Colab 监控脚本（Kaggle 不需要）
├── watchdog.py           # Colab 心跳脚本（Kaggle 不需要）
└── requirements.txt      # 共享依赖清单
```

### 5.3 `train.py` 多平台兼容

训练脚本需要检测运行环境，适配不同的目录结构：

```python
import os
import sys

# 检测运行平台
if os.path.exists("/kaggle/working/"):
    # Kaggle 环境
    DATA_DIR = "/kaggle/input/my-dataset"
    OUTPUT_DIR = "/kaggle/working/output"
    CHECKPOINT_DIR = "/kaggle/working/checkpoints"
elif os.path.exists("/content/"):
    # Colab 环境
    DATA_DIR = "/content/data"
    OUTPUT_DIR = "/content/output"
    CHECKPOINT_DIR = "/content/checkpoints"
else:
    # 本地环境
    DATA_DIR = "./data"
    OUTPUT_DIR = "./output"
    CHECKPOINT_DIR = "./checkpoints"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 其余训练逻辑保持不变...
```

### 5.4 Kaggle 推送工作流

```bash
# 1. 确保数据集已上传
kaggle datasets create -p ./data --dir-mode zip

# 2. 编辑 kernel-metadata.json，确保 dataset_sources 正确

# 3. 推送训练任务
kaggle kernels push -p ./

# 4. 轮询状态（可写入 cron）
kaggle kernels status your-username/my-training-script

# 5. 完成后下载
kaggle kernels output your-username/my-training-script -p ./output
```

### 5.5 多账号策略

考虑到 Kaggle 也是每账号 30 小时/周，可以通过多账号扩展 GPU 时间：

```bash
# 使用不同账号的 API Token
KAGGLE_USERNAME="account1" KAGGLE_KEY="key1" \
  kaggle kernels push -p ./exp1

KAGGLE_USERNAME="account2" KAGGLE_KEY="key2" \
  kaggle kernels push -p ./exp2
```

但需要注意 Kaggle 的 ToS，同一人使用多账号可能违反服务条款。

### 5.6 检查点持久化方案

Kaggle 的检查点管理比 Colab 复杂，因为没有 Google Drive 自动同步：

```
方案 A：保存版本（推荐）
  Save & Run All → 输出文件 → 转换为 Dataset
  下次通过 /kaggle/input/ 读取

方案 B：外部上传（训练结束时）
  在 train.py 末尾添加上传逻辑
  目标：HuggingFace Hub / Google Drive API / HTTP PUT

方案 C：直接下载（最简单）
  kaggle kernels output your-username/my-script -p ./output
  需要轮询等待训练完成
```

### 5.7 完整集成示例

```python
#!/usr/bin/env python3
"""colab-cli style 的 Kaggle 推送工具"""

import subprocess
import json
import time
import sys
from pathlib import Path

KAGGLE_USERNAME = "your-username"


def kaggle_push(project_dir: str, slug: str) -> str:
    """推送训练任务到 Kaggle，返回完整的 kernel slug"""
    result = subprocess.run(
        ["kaggle", "kernels", "push", "-p", project_dir],
        capture_output=True, text=True
    )
    print(result.stdout)
    return f"{KAGGLE_USERNAME}/{slug}"


def kaggle_status(kernel_slug: str) -> str:
    result = subprocess.run(
        ["kaggle", "kernels", "status", kernel_slug],
        capture_output=True, text=True
    )
    return result.stdout.strip()


def wait_for_completion(kernel_slug: str, poll_interval: int = 60):
    """轮询等待训练完成"""
    statuses = {"running", "pending"}
    while True:
        status = kaggle_status(kernel_slug)
        print(f"[{time.strftime('%H:%M:%S')}] {status}")
        if "complete" in status.lower():
            return True
        if "error" in status.lower():
            return False
        time.sleep(poll_interval)


def download_output(kernel_slug: str, output_dir: str):
    subprocess.run(
        ["kaggle", "kernels", "output", kernel_slug, "-p", output_dir],
        check=True
    )


if __name__ == "__main__":
    project = sys.argv[1] if len(sys.argv) > 1 else "."
    slug = "my-training"
    print(f"Pushing {project} to Kaggle...")
    kernel = kaggle_push(project, slug)
    print(f"Kernel: {kernel}")
    print("Waiting for completion (Ctrl+C to detach)...")
    if wait_for_completion(kernel):
        download_output(kernel, "./output")
        print("Done! Output in ./output/")
    else:
        print("Training failed!")
```

---

## 6. 建议与优先级

### 6.1 核心结论

**Kaggle Notebooks 完全可以作为 Colab 的补充 GPU 训练平台**，且在几个关键方面更适合当前的使用场景：

1. **Push 模型解决连接稳定性问题** —— 不依赖 WebSocket 长连接，适合从中国使用代理的环境
2. **配额透明可规划** —— 30 小时/周的明确配额，而非 Colab 的"抽奖"模式
3. **双 GPU 潜力** —— T4 x2 可以运行更大模型
4. **Python Script 原生支持** —— 可以直接运行 `train.py`，无需转换为 notebook

### 6.2 使用建议

| 场景 | 推荐平台 | 原因 |
|------|----------|------|
| 快速原型 / 调试 | **Colab** | 启动更快，无需管理数据集版本 |
| 长时间训练（>4 小时） | **Kaggle** | 不依赖连接稳定性 |
| TPU 工作负载 | **Colab** | Kaggle 不支持 TPU |
| 7B 模型 QLoRA 微调 | **Kaggle** | 双 T4 的 VRAM 优势 |
| 竞赛提交 | **Kaggle** | 天然适合 |
| 批量实验扫描 | **Kaggle** | Push 模型易于自动化 |

### 6.3 实施优先级

**P0 —— 立即可以做的（无需额外开发）**：

1. 为一个现有项目（如 `alexnet_imagenette`）创建 Kaggle 推送配置（`kernel-metadata.json`）
2. 修改 `train.py` 加入多平台检测（检测 `/kaggle/working/` vs `/content/` 目录）
3. 编写 shell 脚本一键推送 + 轮询 + 下载

**P1 —— 短期可以做的**：

1. 在 colab-cli 中添加 `colab kaggle-push` 子命令，封装 `kaggle kernels push`
2. 添加 `colab kaggle-status` 子命令封装 `kaggle kernels status`
3. 添加 `colab kaggle-output` 子命令封装 `kaggle kernels output`
4. 支持多账号配置管理（类似 `colab` 的 `HOME` 切换）

**P2 —— 中期可以考虑的**：

1. 集成现有 MCP 服务器（如 `Seif-Sameh/Kaggle-mcp`），使 Claude Agent 能直接操作 Kaggle
2. 智能调度器：根据当前 Colab 可用性和 Kaggle GPU 队列，自动选择最优平台
3. 检查点自动同步：训练完成时自动从 Kaggle 拉取输出并在本地归档

### 6.4 已知风险

| 风险 | 影响 | 缓解措施 |
|------|------|----------|
| Kaggle 可能收紧免费 GPU 政策 | 失去备用平台 | 同时维护 Colab，不依赖单一平台 |
| 同一人多账号违反 ToS | 账号被封 | 保持合规，仅在合理范围内使用多账号 |
| Kaggle 数据集管理比 Google Drive 繁琐 | 数据流转效率低 | 使用 `kagglehub` 简化，编写辅助脚本 |
| 电话验证从中国操作困难 | 无法开启 GPU | 提前完成验证，尝试 VPN + 手机端 |

### 6.5 与 colab-cli 的整合设计

借鉴 colab-cli 的现有设计模式，新增 Kaggle 子命令：

```
colab kaggle-push <project-dir>           # 推送训练任务
colab kaggle-status [slug]                # 查看状态
colab kaggle-output <slug> [output-dir]   # 下载输出
colab kaggle-list                         # 列出运行中的 Notebook
colab kaggle-dataset-upload <dir>         # 上传数据集
colab kaggle-config                       # 管理多账号 API Token
```

配置文件（`~/.kaggle/config.json`）示例：

```json
{
  "default_account": "main",
  "accounts": {
    "main": {
      "username": "your-username",
      "key": "your-api-key"
    },
    "backup": {
      "username": "backup-username",
      "key": "backup-api-key"
    }
  }
}
```

### 6.6 最终建议

1. **立即试用**：选择一个现有项目，花 30 分钟配置 Kaggle 推送流程，验证端到端可用性
2. **渐进采用**：先在 Kaggle 上运行非关键训练，积累经验后再迁移更重要的任务
3. **两套并行**：保持 Colab 和 Kaggle 两种工作流，Colab 用于快速迭代，Kaggle 用于稳定长时间训练
4. **不要放弃 Colab**：Colab 的 TPU 和 Google Drive 集成在特定场景下仍然不可替代

---

## 参考文献

- [Kaggle Public API Documentation](https://www.kaggle.com/docs/api)
- [Kaggle CLI Cheat Sheet - KDnuggets](https://www.kdnuggets.com/kaggle-cli-cheat-sheet)
- [Kaggle API v1.7.4 发布公告](https://www.kaggle.com/discussions/product-announcements/567753)
- [Git-Driven Kaggle: GitHub Actions + Kaggle API](https://dev.to/yasumorishima/git-driven-kaggle-manage-notebooks-in-github-auto-deploy-via-actions-3ce4)
- [Seif-Sameh/Kaggle-mcp (39 工具 MCP 服务器)](https://github.com/Seif-Sameh/Kaggle-mcp)
- [arrismo/kaggle-mcp (轻量 MCP 服务器)](https://github.com/arrismo/kaggle-mcp)
- [kagglehub PyPI 包](https://pypi.org/project/kagglehub/)
- [Fine-Tune LoRA Models Free on Colab and Kaggle in 2026](https://www.mrcomputerscience.com/fine-tune-lora-models-free-on-colab-and-kaggle-in-2026/)
- [Top Google Colab Alternatives (June 2026)](https://www.thundercompute.com/blog/colab-alternatives-for-cheap-deep-learning-in-2025)
- [Notebooks Session Persistence Update](https://www.kaggle.com/discussions/product-feedback/355440)
