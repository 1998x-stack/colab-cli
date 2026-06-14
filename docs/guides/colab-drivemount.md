# colab drivemount — Google Drive 挂载指南

## 概述

`colab drivemount` 将你的 Google Drive 挂载到 Colab VM 上，就像在浏览器版 Colab 中点击"挂载 Drive"按钮一样。挂载后，Drive 文件在 `/content/drive/MyDrive/` 下可直接读写。

## 基本用法

```bash
# 挂载到默认路径 /content/drive
colab drivemount -s <session-name>

# 挂载到自定义路径
colab drivemount /content/mydrive -s <session-name>
```

默认挂载点：`/content/drive`（即 VM 上的默认路径，与浏览器版 Colab 一致）。

## 工作原理

整个挂载流程分为三层：

### 1. 代码注入

CLI 向 Colab VM 的 kernel 发送一段 Python 代码：

```python
from google.colab import drive
drive.mount('/content/drive')
```

这是 Colab 内置的标准挂载 API，与浏览器版完全相同。

### 2. WebSocket 钩子拦截认证请求

`drive.mount()` 执行时，kernel 会通过 WebSocket 发出一个 `colab_request` 消息（类型为 `dfs_ephemeral`，即 Drive File System 临时凭证请求）。CLI 通过 `ColabRuntime.colab_request_hook` 钩子拦截该消息。

源码位置：`colab_cli/runtime.py:50-87`（`_apply_ws_hook` 方法），它在 WebSocket 的 `on_message` 上挂载拦截器。当 `msg_type == "colab_request"` 时，钩子被触发；若返回 `True`，消息被拦截不再向下传递。

### 3. OAuth 凭证传播

钩子函数（`automation.py:51-127`）执行以下步骤：

1. **向 Colab 后端发起凭证传播请求** → `GET /tun/m/credentials-propagation/{endpoint}`，参数 `authuser=0&authtype=dfs_ephemeral&version=2&dryrun=true&propagate=true`
2. **检查是否需要用户授权** — 若返回 `success: false`，说明该账号尚未授权 Drive 访问：
   - CLI 打印一个 URL 到终端
   - 用户在浏览器中打开 URL，完成 Google OAuth 授权
   - 回到终端按 Enter 继续
3. **正式传播凭证** → 相同端点，`dryrun=false`，携带 `x-goog-colab-token` 头
4. **回复 kernel** → 通过 WebSocket 的 stdin channel 发送 `input_reply` 消息，kernel 收到后继续执行 `drive.mount()` 的剩余逻辑

```
┌──────────────┐     WebSocket      ┌──────────────┐
│   colab CLI  │ ◄─────────────────► │  Colab VM    │
│              │                    │  (kernel)    │
│  drivefs_    │   colab_request    │              │
│  hook()      │ ◄────────────────  │ drive.mount()│
│              │                    │              │
│  OAuth flow  │                    │              │
│  (browser)   │                    │              │
│              │   input_reply      │              │
│              │ ────────────────►  │ mount 完成   │
└──────────────┘                    └──────────────┘
```

## 首次认证流程

首次对某个 Google 账号使用 `colab drivemount` 时，需要一次**交互式浏览器认证**：

1. 运行 `colab drivemount -s mysession`
2. CLI 输出类似以下提示：
   ```
   [colab] REQUIRED: Google Drive Authorization needed.
   Please visit:

   https://accounts.google.com/o/oauth2/auth?...
   ```
3. 在浏览器中打开该 URL，选择 Google 账号并授权
4. 回到终端，按 Enter
5. CLI 显示 `Credentials propagated. Resuming mount...`
6. 挂载完成，`/content/drive/MyDrive/` 可访问

后续挂载同一账号时，由于 OAuth token 已缓存，通常无需再次授权，直接挂载。

## 超时设置

`colab drivemount` 的超时时间为 **600 秒（10 分钟）**（`automation.py:35`），足够完成浏览器认证流程。相比之下，普通 `colab exec` 的默认超时仅为 10 秒。

## 限制与注意事项

1. **必须已有一个运行中的 Colab session** — `colab drivemount` 不创建新 session，需要先用 `colab new` 创建。
2. **首次需要浏览器交互** — 无法完全静默 / 无人值守。若需自动化，可预先在浏览器版 Colab 中完成 Drive 授权（同一账号后续 CLI 挂载即免交互）。
3. **不支持服务账号（ADC）认证的 Drive 挂载** — Drive 挂载需要 `userinfo.email` 和 `drive` scope，ADC（Application Default Credentials）通常不具备这些 scope。
4. **挂载的是个人 Drive** — `/content/drive/MyDrive/` 是用户的个人 Drive。共享 Drive（Shared Drives）需额外配置。
5. **VM 停止后挂载自动解除** — Drive 挂载是临时的，session 结束后数据不保留在 VM 上（Drive 上的文件不受影响）。
6. **`drive.mount()` 内部超时 120 秒** — 从 CLI 发送 Enter（credential propagation）后开始计时。若 OAuth 授权未在 120 秒内完成，kernel 端抛出 `ValueError: mount failed`。务必在浏览器弹出后尽快授权。
7. **CPU session 即可挂载** — Drive 挂载不需要 GPU。若仅需挂载后访问文件（不训练），用 `colab new -s <name>` 创建 CPU session 即可，不占用 GPU 配额。

## 实战测试记录

**测试日期：** 2026-06-14
**测试环境：** macOS, 中国（通过代理）, colab CLI v0.5.11

### 测试 1：基础挂载流程

```bash
colab new -s drive-test          # CPU session（不需要 GPU）
colab drivemount -s drive-test   # 触发 OAuth 流程
```

**结果：** 成功拦截 `dfs_ephemeral` 认证请求，OAuth URL 正确生成，浏览器打开后需手动授权。

**关键时间节点：**
| 时间 | 事件 |
|------|------|
| T+0s | `colab drivemount` 发送 `drive.mount()` 代码 |
| T+3s | kernel 发出 `colab_request`（`dfs_ephemeral`） |
| T+5s | CLI 打印 OAuth URL，等待用户按 Enter |
| 用户按 Enter 后 | CLI 调用 `credentials-propagation` API |
| propagation 后 120s 内 | kernel 端 `drive.mount()` 等待 DFS 凭证就绪 |
| 超时 | `ValueError: mount failed` |

### 测试 2：认证未完成时的失败模式

若浏览器打开了 OAuth URL 但用户未点击"允许"：
- CLI 端正常返回（exit 0）
- kernel 端 `drive.mount()` 在 120s 后抛出 `ValueError: mount failed`
- session 状态变为 BUSY (automation(drivemount))，无法复用

**教训：** 必须在 120 秒内完成浏览器授权。窗口很短，建议先准备好浏览器登录状态。

### 测试 3：进程中断后的 session 状态

若 `colab drivemount` 进程被 `kill`（如 Ctrl+C 或 SIGTERM）：
- kernel 仍在执行 `drive.mount()`，等待已断开的 WebSocket 回复
- session 状态显示 `BUSY (automation(drivemount))`
- **该 session 无法再次挂载** — 需 `colab stop -s <name>` + `colab new` 重建

**教训：** 不要中途 kill drivemount 进程。若已 kill，必须重建 session。

## 自动化方案

首次挂载需要浏览器交互，无法完全无人值守。但可以通过以下方案减少手动操作：

### 方案 A：预授权（推荐）

在浏览器版 Colab 中先完成一次 Drive 挂载授权。之后同一账号的 CLI 挂载无需再次授权——OAuth token 由 Colab 后端缓存，`credentials-propagation` 直接成功。

```bash
# 1. 打开浏览器 Colab，新建 notebook，点击"挂载 Drive"按钮
# 2. 完成 OAuth 授权
# 3. 回到终端：
colab new -s training
colab drivemount -s training  # 直接成功，无需浏览器
```

### 方案 B：自动化 wrapper（首次或 token 过期时）

以下 Python wrapper 自动捕获 OAuth URL、打开浏览器、等待授权完成后继续：

```python
#!/usr/bin/env python3
"""Wrapper: 自动打开浏览器完成 Drive OAuth，然后继续挂载。"""
import subprocess, re, sys, os, time, threading

os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7890")
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7890")
os.environ.setdefault("ALL_PROXY", "socks5://127.0.0.1:7890")

# 注意：drive.mount() 内部超时 120s，所以这里的等待不能超过 120s
WAIT_SECONDS = 90  # 给用户留 90s 完成浏览器授权

proc = subprocess.Popen(
    ["colab", "drivemount"] + sys.argv[1:],
    stdin=subprocess.PIPE, stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT, text=True, bufsize=1,
)

url_pattern = re.compile(r'https://accounts\.google\.com/o/oauth2/[^\s]+')
auth_url = None

def reader():
    global auth_url
    for line in iter(proc.stdout.readline, ""):
        sys.stdout.write(line); sys.stdout.flush()
        if auth_url is None:
            m = url_pattern.search(line)
            if m: auth_url = m.group(0).rstrip(".")

t = threading.Thread(target=reader, daemon=True); t.start()

while auth_url is None and t.is_alive():
    time.sleep(0.5)

if auth_url:
    subprocess.run(["open", auth_url], timeout=5)
    print(f"[wrapper] 请在浏览器中完成授权（{WAIT_SECONDS}s 超时）...")
    time.sleep(WAIT_SECONDS)
    proc.stdin.write("\n"); proc.stdin.flush()

t.join(timeout=30)
proc.wait()
```

**关键实现细节：**
- **必须使用线程分离 stdout 读取** — `drive.mount()` 的 `input()` 提示符无尾部换行，`readline()` 会永久阻塞。后台线程持续读取，主线程通过正则匹配检测 URL
- **等待时间 < 120s** — wrapper 的等待时间必须小于 `drive.mount()` 的内部超时（120s），否则授权完成也无济于事
- **代理配置** — wrapper 需要与 `colab` CLI 相同的代理环境变量

## 实用工作流

```bash
# 1. 创建 session
colab new --gpu T4 -s training

# 2. 挂载 Drive
colab drivemount -s training

# 3. 查看 Drive 文件
colab exec -s training -c "import os; print(os.listdir('/content/drive/MyDrive/'))"

# 4. 从 Drive 读取数据训练，将 checkpoint 写回 Drive
colab upload train.py /content/train.py
colab exec -f train.py --timeout 3600

# 5. 训练完成后，checkpoint 已在 Drive 中，直接停掉 VM
colab stop -s training
```

## 源码关键路径

| 文件 | 位置 | 作用 |
|------|------|------|
| `commands/automation.py:192-211` | `drivemount()` 函数 | CLI 入口，构造挂载代码，调用 `run_automation()` |
| `commands/automation.py:38-169` | `run_automation()` | 注册 `drivefs_hook`，通过 runtime 执行代码 |
| `commands/automation.py:51-127` | `drivefs_hook()` | 拦截 `dfs_ephemeral` 认证请求，完成 OAuth 流程 |
| `runtime.py:46-87` | `_apply_ws_hook()` | WebSocket 消息拦截器，将 `colab_request` 路由到钩子 |
| `runtime.py:164-246` | `execute_code()` | 执行 kernel 代码，支持 stdin hook 和自定义超时 |

## 与浏览器版 Colab 的对比

| 特性 | 浏览器 Colab | colab drivemount |
|------|-------------|------------------|
| 挂载方式 | 点击侧边栏按钮 | `colab drivemount -s <name>` |
| 认证 | 弹窗 OAuth | 终端 URL + 浏览器 OAuth |
| 挂载点 | `/content/drive` | `/content/drive`（可自定义） |
| 后续挂载 | 一键挂载 | 免交互（token 已缓存） |
| 文件访问 | 侧边栏 + 代码 | 代码 + `colab exec` / `colab download` |
