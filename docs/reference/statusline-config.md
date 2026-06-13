# Claude Code Status Line 配置指南

> 深入理解 status line 的 JSON 输入结构、进度条实现原理与自定义配置方法。

---

## 目录

1. [概述](#1-概述)
2. [配置方式](#2-配置方式)
3. [JSON 输入结构全解](#3-json-输入结构全解)
4. [进度条实现](#4-进度条实现)
5. [完整示例](#5-完整示例)
6. [颜色与样式](#6-颜色与样式)
7. [常见问题](#7-常见问题)

---

## 1. 概述

Status line 是 Claude Code 终端底部的状态栏，每次工具调用后自动刷新。它通过 stdin 接收 JSON 数据，脚本输出纯文本（可带 ANSI 转义序列）显示在终端底部。

**核心机制：**
- 输入：stdin，完整 JSON，包含工作区、模型、上下文窗口、成本等信息
- 输出：stdout，每行一个状态条目（Claude Code 渲染为状态栏）
- 刷新频率：每次 API 调用 / 工具执行后
- 超时：脚本执行超过 500ms 会被 kill，状态栏显示 fallback

## 2. 配置方式

在 `~/.claude/settings.json` 中配置：

```json
{
  "statusLine": {
    "type": "command",
    "command": "bash /Users/<user>/.claude/statusline-command.sh"
  }
}
```

`type` 可选值：
- `"command"` — 执行 shell 命令，stdin 传入 JSON
- `"text"` — 直接显示静态文本

推荐使用独立脚本文件而非内联命令，方便维护和调试。

### 调试技巧

手动测试脚本输出：

```bash
echo '{"workspace":{"current_dir":"/path/to/project"},"model":{"display_name":"Opus"},"context_window":{"used_percentage":45}}' | bash ~/.claude/statusline-command.sh
```

## 3. JSON 输入结构全解

### 3.1 顶层字段

| 字段 | 类型 | 说明 |
|------|------|------|
| `cwd` | string | 当前工作目录 |
| `session_id` | string | 会话唯一标识 |
| `session_name` | string | 自定义会话名（`--name` 设置） |
| `transcript_path` | string | 对话记录文件路径 |
| `version` | string | Claude Code 版本号 |
| `exceeds_200k_tokens` | bool | 上次响应是否超过 200k token |

### 3.2 模型信息 (`model`)

| 字段 | 类型 | 说明 |
|------|------|------|
| `model.id` | string | 模型 ID，如 `claude-opus-4-7` |
| `model.display_name` | string | 显示名：`Opus` / `Sonnet` / `Haiku` |

### 3.3 工作区 (`workspace`)

| 字段 | 类型 | 说明 |
|------|------|------|
| `workspace.current_dir` | string | 当前目录 |
| `workspace.project_dir` | string | Claude Code 启动目录 |
| `workspace.added_dirs` | string[] | `/add-dir` 添加的目录 |
| `workspace.git_worktree` | string | git worktree 名称（仅 worktree 模式） |
| `workspace.repo.host` | string | 远程仓库 host，如 `github.com` |
| `workspace.repo.owner` | string | 仓库所有者 |
| `workspace.repo.name` | string | 仓库名称 |

### 3.4 上下文窗口 (`context_window`) — 进度条核心数据

| 字段 | 类型 | 说明 |
|------|------|------|
| `context_window.total_input_tokens` | int | 当前输入 token 数 |
| `context_window.total_output_tokens` | int | 当前输出 token 数 |
| `context_window.context_window_size` | int | 最大上下文窗口（默认 200000） |
| `context_window.used_percentage` | int | **已用百分比（0-100）** |
| `context_window.remaining_percentage` | int | **剩余百分比（0-100）** |
| `context_window.current_usage.input_tokens` | int | 最近一次请求的输入 token |
| `context_window.current_usage.output_tokens` | int | 最近一次请求的输出 token |
| `context_window.current_usage.cache_creation_input_tokens` | int | 缓存创建 token |
| `context_window.current_usage.cache_read_input_tokens` | int | 缓存命中 token |

> `used_percentage` 是最适合做进度条的字段——它实时反映上下文窗口消耗，是会话的"燃料表"。

### 3.5 成本与会话时长 (`cost`)

| 字段 | 类型 | 说明 |
|------|------|------|
| `cost.total_cost_usd` | float | 本次会话估算费用（美元） |
| `cost.total_duration_ms` | int | 会话墙钟时间（毫秒） |
| `cost.total_api_duration_ms` | int | API 等待时间（毫秒） |
| `cost.total_lines_added` | int | 累计新增代码行数 |
| `cost.total_lines_removed` | int | 累计删除代码行数 |

### 3.6 速率限制 (`rate_limits`)

| 字段 | 类型 | 说明 |
|------|------|------|
| `rate_limits.five_hour.used_percentage` | float | 5 小时窗口使用百分比 |
| `rate_limits.five_hour.resets_at` | int | 重置时间（Unix 时间戳） |
| `rate_limits.seven_day.used_percentage` | float | 7 天窗口使用百分比 |
| `rate_limits.seven_day.resets_at` | int | 重置时间（Unix 时间戳） |

### 3.7 条件字段（仅在对应功能激活时出现）

| 字段 | 类型 | 说明 |
|------|------|------|
| `effort.level` | string | 思考强度：`low` / `medium` / `high` / `xhigh` / `max` |
| `thinking.enabled` | bool | 是否启用扩展思考 |
| `output_style.name` | string | 输出风格名称 |
| `vim.mode` | string | Vim 模式：`NORMAL` / `INSERT` / `VISUAL` / `VISUAL LINE` |
| `agent.name` | string | 当前 agent 名称（`--agent` 模式） |
| `pr.number` | int | 当前分支的开放 PR 编号 |
| `pr.url` | string | PR URL |
| `pr.review_state` | string | 审查状态：`approved` / `pending` / `changes_requested` / `draft` |

### 3.8 Worktree 字段（仅 `--worktree` 模式）

| 字段 | 类型 | 说明 |
|------|------|------|
| `worktree.name` | string | worktree 名称 |
| `worktree.path` | string | worktree 绝对路径 |
| `worktree.branch` | string | 关联分支 |
| `worktree.original_cwd` | string | 进入 worktree 前的目录 |
| `worktree.original_branch` | string | 进入 worktree 前的分支 |

## 4. 进度条实现

### 4.1 设计思路

上下文窗口是会话中最重要的资源。随着对话进行，历史消息不断累积，上下文窗口逐渐填满。`context_window.used_percentage` 天然适合作为进度条——它是线性的、单调递增的、有明确的上限（100%）。

### 4.2 实现代码

```bash
#!/bin/bash
input=$(cat)

# 提取已用百分比
used=$(echo "$input" | jq -r '.context_window.used_percentage // empty')

if [ -n "$used" ] && [ "$used" != "null" ]; then
  used_int=$(printf "%.0f" "$used")
  bar_width=10
  filled=$(( used_int * bar_width / 100 ))
  empty=$(( bar_width - filled ))

  # 颜色策略：<50% 绿，50-80% 黄，>80% 红
  if [ "$used_int" -lt 50 ]; then
    bar_color='\033[32m'   # green
  elif [ "$used_int" -lt 80 ]; then
    bar_color='\033[33m'   # yellow
  else
    bar_color='\033[31m'   # red
  fi

  # 渲染：ctx ████░░░░░░  45%
  printf ' \033[90mctx\033[0m '
  printf "${bar_color}"
  for ((i=0; i<filled; i++)); do printf '█'; done
  printf '\033[0m\033[90m'
  for ((i=0; i<empty; i++)); do printf '░'; done
  printf '\033[0m'
  printf ' \033[90m%3d%%\033[0m' "$used_int"
fi
```

### 4.3 视觉效果

| 使用率 | 显示效果 |
|--------|---------|
| 低于 50% | `ctx ████░░░░░░  35%` (绿色) |
| 50-80% | `ctx ███████░░░  72%` (黄色) |
| 超过 80% | `ctx ████████░░  88%` (红色) |

> 红色时建议使用 `/compact` 压缩上下文，释放空间。

### 4.4 进阶：多段进度条

可以组合多个指标形成复合进度条：

```bash
# 上下文窗口 + 速率限制
rate_5h=$(echo "$input" | jq -r '.rate_limits.five_hour.used_percentage // 0')
printf ' ctx:%d%% rate5h:%d%%' "$used_int" "${rate_5h%.*}"
```

## 5. 完整示例

以下是一个完整的 status line 脚本，融合 robbyrussell 主题风格 + 上下文进度条：

```bash
#!/bin/bash
input=$(cat)

# --- 目录信息 ---
cwd=$(echo "$input" | jq -r '.workspace.current_dir // empty')
[ -z "$cwd" ] && cwd=$(echo "$input" | jq -r '.workspace.project_dir // "~"')
dir_short=$(basename "$cwd" 2>/dev/null || echo "$cwd")

# --- robbyrussell 风格：➜ 目录名 ---
printf '\033[32m➜\033[0m \033[36m%s\033[0m' "$dir_short"

# --- git 分支与状态 ---
branch=$(cd "$cwd" 2>/dev/null && GIT_OPTIONAL_LOCKS=0 git symbolic-ref --short HEAD 2>/dev/null)
if [ -n "$branch" ]; then
  status=$(cd "$cwd" 2>/dev/null && GIT_OPTIONAL_LOCKS=0 git status --porcelain 2>/dev/null)
  if [ -n "$status" ]; then
    printf ' \033[34mgit:(\033[31m%s\033[34m) \033[33m✗\033[0m' "$branch"
  else
    printf ' \033[34mgit:(\033[31m%s\033[34m)\033[0m' "$branch"
  fi
fi

# --- 上下文进度条 ---
used=$(echo "$input" | jq -r '.context_window.used_percentage // empty')
if [ -n "$used" ] && [ "$used" != "null" ]; then
  used_int=$(printf "%.0f" "$used")
  bar_width=10
  filled=$(( used_int * bar_width / 100 ))
  empty=$(( bar_width - filled ))

  if [ "$used_int" -lt 50 ]; then
    bar_color='\033[32m'
  elif [ "$used_int" -lt 80 ]; then
    bar_color='\033[33m'
  else
    bar_color='\033[31m'
  fi

  printf ' \033[90mctx\033[0m '
  printf "${bar_color}"
  for ((i=0; i<filled; i++)); do printf '█'; done
  printf '\033[0m\033[90m'
  for ((i=0; i<empty; i++)); do printf '░'; done
  printf '\033[0m'
  printf ' \033[90m%3d%%\033[0m' "$used_int"
fi

# --- 模型名称 ---
model=$(echo "$input" | jq -r '.model.display_name // empty')
if [ -n "$model" ] && [ "$model" != "null" ]; then
  printf ' \033[90m%s\033[0m' "$model"
fi

echo
```

### 效果预览

```
➜ colab-cli git:(main) ✗ ctx ████░░░░░░  45% Opus
➜ nanochat-colab git:(feat/train) ctx ███████░░░  72% Sonnet
➜ transformer_iwslt git:(main) ctx ██████████ 100% Opus
```

## 6. 颜色与样式

### 6.1 ANSI 颜色速查

| 代码 | 颜色 | 用途建议 |
|------|------|---------|
| `\033[32m` | 绿色 | 正常状态、低占用 |
| `\033[33m` | 黄色 | 警告状态、中等占用 |
| `\033[31m` | 红色 | 危险状态、高占用 |
| `\033[34m` | 蓝色 | git 信息 |
| `\033[36m` | 青色 | 路径、目录 |
| `\033[90m` | 暗灰 | 辅助信息（标签、百分比） |
| `\033[0m` | 重置 | 每个颜色段落后必须重置 |

### 6.2 性能注意事项

- **不要调用外部 API**：脚本需在 500ms 内完成，否则被 kill
- **减少子进程**：`git` 命令已经很快，但避免在循环中反复调用
- **缓存重型计算**：如需计算 git stash 数量等，考虑写到临时文件
- **`GIT_OPTIONAL_LOCKS=0`**：防止 status line 的 git 调用阻塞其他 git 操作

### 6.3 必装依赖

脚本依赖 `jq` 解析 JSON。macOS 安装：

```bash
brew install jq
```

## 7. 常见问题

### Q: 状态栏不显示？

检查脚本是否可执行、`jq` 是否安装、JSON 路径是否正确。手动运行脚本加 echo 调试。

### Q: 进度条不更新？

确认 `context_window.used_percentage` 字段存在。仅在工具调用后刷新——如果长时间没有工具调用，进度条不会变化。

### Q: 中文乱码？

ANSI 转义序列与 UTF-8 兼容。确保终端支持 UTF-8，脚本使用 UTF-8 编码。

### Q: 如何隐藏某些信息？

在脚本中注释掉对应 `printf` 行即可。status line 应保持简洁，建议不超过一行。

### Q: 脚本执行太慢被 kill？

- 用 `jq` 一次提取多个字段：`jq -r '.model.display_name, .context_window.used_percentage'`
- 避免 `awk` / `sed` 多管道串联处理
- 避免 `curl` 或网络请求
- git 命令加上 `--no-optional-locks` 或 `GIT_OPTIONAL_LOCKS=0`

---

> 关键原则：status line 是"仪表盘"而非"日志"——只显示当前最关键的状态信息，保持简洁、快速、易读。
