# Claude Code Auto Compaction 深度解析

> 剖析 Claude Code 上下文压缩的内部机制、硬编码 prompt 结构，以及 5 种注入自定义指令的实战方案。

---

## 目录

1. [概述](#1-概述)
2. [Compaction 触发机制](#2-compaction-触发机制)
3. [硬编码 Prompt 全解](#3-硬编码-prompt-全解)
4. [5 种注入自定义指令的方法](#4-5-种注入自定义指令的方法)
5. [最佳实践](#5-最佳实践)
6. [已知问题与陷阱](#6-已知问题与陷阱)

---

## 1. 概述

当对话上下文接近 200k token 上限时，Claude Code 自动触发 compaction——用一个 LLM 调用将完整对话历史压缩为结构化摘要，释放上下文窗口空间。

**核心事实：**

- compaction 的 system prompt **硬编码在二进制中**，无法通过 `settings.json` 替换
- compaction 使用的模型与主会话相同，但禁用工具调用和扩展思考（`maxTurns: 1`）
- API 调用带有 `querySource: "compact"` 标记，跳过缓存写入（`SkipCacheWrite: true`）
- 共有三种变体：全量 compact、部分 compact（from）、部分 compact（up_to）

## 2. Compaction 触发机制

### 2.1 自动触发

当上下文 token 使用率达到阈值时自动触发。默认阈值内建于二进制，可通过环境变量覆盖：

```bash
export CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=75  # 使用率达到 75% 时触发
```

相关 settings.json 开关：

```json
{
  "autoCompactEnabled": true
}
```

### 2.2 手动触发

```bash
/compact                          # 默认全量压缩
/compact focus on auth module     # 指定保留重点
/compact 保留数据库schema和API路由  # 中文指令同样有效
```

手动 compact 的 free-text 参数会传入 compaction 过程，影响模型生成摘要时的优先级。

### 2.3 三种 Compaction 变体

| 变体 | 内部名 | 行为 |
|------|--------|------|
| 全量压缩 | `BASE_COMPACT` | 总结整个对话历史 |
| 部分压缩（from） | `PARTIAL_COMPACT` | 从指定点向后压缩，保留之前的消息和缓存前缀 |
| 部分压缩（up_to） | `PARTIAL_COMPACT_UP_TO` | 压缩指定点之前的消息，保留最近消息原样 |

> 部分压缩利用 prompt cache 前缀复用降低成本，是自动 compaction 的默认策略。

## 3. 硬编码 Prompt 全解

### 3.1 整体结构

prompt 以双段"禁止工具调用"的警告开头和结尾：

```
CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.
- Do NOT use Read, Bash, Grep, Glob, Edit, Write, or ANY other tool.
- Tool calls will be REJECTED and will waste your only turn — you will fail the task.
- Your entire response must be plain text: an <analysis> block followed by a <summary> block.
```

模型输出两个 XML 块：

- `<analysis>` — 思考草稿，提取后**丢弃**，不保留
- `<summary>` — 实际保留的摘要，包含 9 个固定章节

### 3.2 9 个必填章节

| # | 章节 | 保留内容 | 为什么重要 |
|---|------|---------|-----------|
| 1 | **Primary Request and Intent** | 所有用户显式请求及细节 | 不丢失用户原始意图 |
| 2 | **Key Technical Concepts** | 技术栈、框架、架构决策 | 后续回复的技术语境 |
| 3 | **Files and Code Sections** | 所有检查/修改/创建的文件，含完整代码片段和理由 | 代码改动的完整记录 |
| 4 | **Errors and Fixes** | 每个错误及其解决方式 | 避免重复踩坑 |
| 5 | **Problem Solving** | 已解决的问题及正在进行的方法 | 工作连续性 |
| 6 | **All User Messages** | 逐字保留所有用户消息（含纠正和偏好） | 这是最关键的章节——用户反馈和偏好不容易在压缩中丢失 |
| 7 | **Pending Tasks** | 明确请求但未完成的任务 | 确保不遗漏未完成工作 |
| 8 | **Current Work** | 压缩前正在做的精确工作，含文件名和代码 | 恢复上下文后能立即继续 |
| 9 | **Optional Next Step** | 建议的下一步，引用对话原文作依据 | 引导恢复后的第一个动作 |

### 3.3 设计意图

这个 9 段结构旨在回答："如果我是一个新会话，需要知道什么才能无缝继续？"

第 6 章（All User Messages）最关键——它确保用户的纠正、偏好变更、明确否定等不会被压缩过程"平滑"掉。

## 4. 5 种注入自定义指令的方法

### 方法一：PreCompact Hook（直接注入）

**原理：** PreCompact hook 的 stdout 在 compaction 调用时被直接注入到请求中。

```json
{
  "hooks": {
    "PreCompact": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "cat ~/.claude/compact-instructions.md"
          }
        ]
      }
    ]
  }
}
```

自定义指令文件示例（`~/.claude/compact-instructions.md`）：

```markdown
IMPORTANT compaction priorities for this project:
- ALWAYS preserve the full database schema (projects/backend/prisma/schema.prisma)
- Preserve all auth-related decisions and middleware changes
- Keep error messages and their fixes verbatim
- Retain all user corrections about code style preferences
- The project uses Python 3.11+ with FastAPI + SQLAlchemy
```

**支持四种 hook 类型：**

| 类型 | 说明 | 适用场景 |
|------|------|---------|
| `command` | 执行 shell 命令，stdout 注入 | 读取静态文件、运行脚本 |
| `agent` | 派生子 agent，可读取/搜索项目文件 | 动态提取项目上下文 |
| `prompt` | 调用 LLM（默认 Haiku），结果注入 | 智能提取关键信息 |
| `http` | 调用远程 API（需配 `allowedHttpHookUrls`） | 外部系统集成 |

**阻断 compaction：** hook 返回 exit code 2 可完全阻止 compaction（但上下文最终会耗尽，慎用）。

**注意：** GitHub issue #50467 报告 PreCompact 在自动 compaction 时可能不触发。详见第 6 节。

### 方法二：手动 `/compact` 指令

**原理：** free-text 参数传入 compaction 调用，影响摘要内容的优先级。

```bash
/compact focus on the auth module and database schema
/compact 保留认证模块和多账户配置，忽略测试文件
/compact keep error fixes verbatim, summarize exploration only
```

**优点：** 精准、可用、无配置成本。
**缺点：** 仅手动触发，无法自动化。

### 方法三：CLAUDE.md（可靠存活）

**原理：** compaction 后，系统从磁盘重新读取 CLAUDE.md，因此其内容天然存活。

```markdown
# CLAUDE.md

## Compaction Survival Notes
以下信息必须在 compaction 后依然可用：

- 数据库连接串配置在 .env 中，不要硬编码
- 用户偏好：异步 > 同步，pydantic v2 风格
- 当前分支 feature/multi-account 的目标是支持三个账号轮换
```

**这是目前最可靠的保持关键信息跨 compaction 存活的方式。** 约束是 CLAUDE.md 会被每次对话完整加载，所以要精简。

### 方法四：Auto Memory（MEMORY.md）

**原理：** Claude Code v2.1.59+ 的 auto memory 功能独立于对话上下文，compaction 后重新加载。

Memory 文件位于 `~/.claude/projects/<project>/memory/`，按类型组织：

```markdown
# MEMORY.md（索引文件）

- [多账户架构决策](memory/multi-account-arch.md) — Colab 多账户轮换的关键设计决策
- [vLLM 版本兼容](memory/vllm-compat.md) — T4 上的 vLLM 版本选择与 monkey-patch 方案
```

与 CLAUDE.md 的区别：memory 文件是 Claude 自动维护的，可以按主题拆分，按需加载。

### 方法五：Stop Hook + 状态文件（变通方案）

**原理：** 在每轮对话后写入状态文件，结合 CLAUDE.md 指示在回复前读取。

```json
{
  "hooks": {
    "Stop": [
      {
        "type": "command",
        "command": "python3 ~/.claude/save-compaction-state.py"
      }
    ]
  }
}
```

状态保存脚本示例：

```python
#!/usr/bin/env python3
"""每轮对话后保存关键状态，供 compaction 恢复使用。"""
import json, os, time

state_file = os.path.expanduser("~/.claude/compaction-state.json")
now = time.time()

# 节流：30 秒内不重复写入
try:
    mtime = os.path.getmtime(state_file)
    if now - mtime < 30:
        exit(0)
except FileNotFoundError:
    pass

state = {
    "timestamp": now,
    "current_branch": os.popen("git branch --show-current").read().strip(),
    "modified_files": os.popen("git diff --name-only").read().strip().split("\n"),
}

with open(state_file, "w") as f:
    json.dump(state, f, indent=2)
```

配合 CLAUDE.md：

```markdown
## Compaction Recovery
Before responding after compaction, read ~/.claude/compaction-state.json
to restore: current branch, modified files, and in-progress task context.
```

### 方法对比总览

| 方法 | 自动化 | 注入到 prompt | 跨 compaction 存活 | 可靠性 |
|------|--------|--------------|-------------------|--------|
| PreCompact Hook | 是 | 是（stdout 注入） | — | **有 bug**（见 §6） |
| `/compact focus` | 否（手动） | 是（参数传入） | — | 始终可用 |
| CLAUDE.md | 是 | 否（磁盘重读） | 是 | **最可靠** |
| Auto Memory | 是 | 否（磁盘重读） | 是 | 可靠 |
| Stop Hook + 状态文件 | 是 | 否（磁盘重读） | 是 | 可靠（需节流） |

## 5. 最佳实践

### 5.1 推荐组合策略

```
PreCompact Hook（如果版本支持）
  +
CLAUDE.md 关键信息
  +
Auto Memory 自动维护
```

### 5.2 CLAUDE.md 编写原则

compaction 后重读的 CLAUDE.md 是最后一道防线。应包含：

- **项目技术栈和版本约束**（Python 版本、关键依赖）
- **当前分支目标和上下文**
- **用户偏好和约定**（代码风格、架构偏好）
- **已知陷阱和避坑指南**

不应包含：

- 临时状态（用 memory 或状态文件）
- 过长的代码片段（compaction prompt 本身会保留）
- 可以在 git log 中找到的信息

### 5.3 验证 compaction 效果

```bash
# 查看 compaction 后的上下文大小变化
# 在 compaction 前后对比 context_window.total_input_tokens

# 手动触发测试
/compact
# 然后问 Claude："刚才 compaction 保留了哪些关键信息？"
```

## 6. 已知问题与陷阱

### 6.1 PreCompact Hook 不触发（Issue #50467）

**症状：** PreCompact hook 在自动 compaction 时不触发，手动 `/compact` 也可能受影响。

**影响版本：** v2.1.105 - v2.1.114（可能更广）。

**变通方案：**

1. 使用 Stop hook 替代（详见方法五）
2. 依赖 CLAUDE.md + MEMORY.md 作为主要存活机制
3. 定期手动 `/compact focus on ...` 确认状态

### 6.2 中文内容在 compaction 中的表现

compaction prompt 是英文的，但模型能正确处理中文对话。关键章节（尤其是"All User Messages"）会逐字保留中文原文。不过：

- 技术术语可能被翻译——在 CLAUDE.md 中明确"保留中文术语"
- 代码注释中的中文通常被完整保留

### 6.3 性能考虑

- compaction 调用**不计入速率限制**（`querySource: "compact"`）
- 使用 prompt cache 前缀复用降低成本
- 禁用了扩展思考以减少延迟
- PreCompact hook 中的 `agent` 类型会额外消耗 token（用 Haiku 执行）

### 6.4 不要过度依赖单一机制

compaction 是一个尽力而为的过程——没有任何方案能保证 100% 的信息保真度。最佳策略是**多层防御**：CLAUDE.md（基础）+ PreCompact hook（增强）+ 定期验证。

---

> 核心认知：compaction 不是归档，是压缩——信息密度提高，但信息量必然减少。关键不是阻止丢失，而是确保丢失的内容不致命。
