# DeepSeek-Colab CLI 深度使用指南

> 基于 `google-colab-cli v0.5.9` 源码级逆向分析，揭示官方文档未覆盖的内部机制、设计权衡与实战陷阱。

---

## 目录

1. [架构总览](#1-架构总览)
2. [认证体系深度解析](#2-认证体系深度解析)
3. [会话生命周期](#3-会话生命周期)
4. [Keep-Alive 守护进程](#4-keep-alive-守护进程)
5. [代码执行机制](#5-代码执行机制)
6. [文件传输协议](#6-文件传输协议)
7. [colab run：一次性任务引擎](#7-colab-run一次性任务引擎)
8. [自动化命令内部机制](#8-自动化命令内部机制)
9. [状态持久化与锁机制](#9-状态持久化与锁机制)
10. [更新检测系统](#10-更新检测系统)
11. [实战陷阱与排错手册](#11-实战陷阱与排错手册)
12. [命令行完整参考](#12-命令行完整参考)

---

## 1. 架构总览

### 1.1 模块结构

```
colab_cli/
├── cli.py              # 入口：Typer CLI 定义，全局回调，子命令注册
├── client.py           # Colab API 客户端：分配/释放 VM、keep-alive RPC
├── runtime.py          # ColabRuntime：封装 jupyter-kernel-client 的远程执行
├── auth.py             # 双认证通道：OAuth2（InstalledAppFlow）与 ADC
├── state.py            # 状态持久化：SessionState、Settings，fcntl 文件锁
├── common.py           # 全局 State 单例，会话同步，keep-alive 进程管理
├── contents.py         # 文件 CRUD：基于 Jupyter Contents API
├── console.py          # 原始 TTY：WebSocket 连接 /colab/tty 端点
├── repl.py             # 交互式 REPL：prompt_toolkit + Pygments
├── history.py          # JSONL 事件日志：执行记录、文件操作、keep-alive 状态
├── converter.py        # 日志导出：.ipynb / .md / .jsonl
├── auto_update.py      # 版本检测 + 自升级
├── utils.py            # 工具函数：状态码提取、Kitty 终端图像渲染
└── commands/
    ├── session.py      # new, sessions, status, stop, restart-kernel, keep-alive
    ├── execution.py    # exec, repl, console
    ├── files.py        # ls, rm, upload, download, edit
    ├── automation.py   # auth, drivemount, install
    ├── run.py          # run（一次性任务）
    └── utility.py      # pay, log, url, version, update, whoami, readme, skill
```

### 1.2 核心数据流

```
用户 → Typer CLI → State 单例 → Client (HTTP) → Colab API
                                  → ColabRuntime (WebSocket) → Jupyter Kernel
                                  → StateStore (fcntl JSON) → ~/.config/colab-cli/
                                  → HistoryLogger (JSONL) → ~/.config/colab-cli/history/
```

### 1.3 关键外部依赖

| 依赖 | 作用 |
|------|------|
| `typer` + `click` | CLI 框架，`allow_extra_args` 支持 shebang 传参 |
| `jupyter-kernel-client` | WebSocket 连接远程 Jupyter Kernel |
| `google-auth-oauthlib` | OAuth2 InstalledAppFlow 流程 |
| `google-auth` | ADC 认证 + Token 刷新 |
| `requests` + `websocket-client` | HTTP + WebSocket 传输 |
| `prompt-toolkit` + `pygments` | 交互式 REPL |
| `pydantic` | 数据模型与 API 响应校验 |
| `fcntl` | 文件锁，保证多进程并发安全 |

---

## 2. 认证体系深度解析

### 2.1 双通道设计

CLI 支持两种认证策略，通过全局 `--auth` 标志切换：

```bash
colab --auth oauth2 new          # OAuth2（默认）
colab --auth adc new             # Application Default Credentials
```

### 2.2 OAuth2 流程（默认）

**源码路径：** `auth.py:_get_google_auth_credentials()`

1. 从 `~/.colab-cli-oauth-config.json`（或内置 `oauth_config.json`）加载客户端配置
2. 检查 `~/.config/colab-cli/token.json` 是否有已保存的 token
3. 若 token 有效 → 直接使用
4. 若 token 过期但有 refresh_token → 尝试刷新
5. 若无法刷新 → 启动本地 OAuth 服务器（端口 **8200**），触发浏览器授权流程
6. 成功后保存 token 到 `~/.config/colab-cli/token.json`

**请求的 OAuth Scopes：**
```python
PUBLIC_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/colaboratory",   # keep-alive RPC 必需
    "https://www.googleapis.com/auth/drive.file",
]
```

### 2.3 ADC 流程

**源码路径：** `auth.py:_get_adc_credentials()`

1. 调用 `google.auth.default(scopes=PUBLIC_SCOPES)`
2. 按标准 ADC 发现链查找凭据：`GOOGLE_APPLICATION_CREDENTIALS` → `gcloud auth application-default login` → GCE/GKE metadata server
3. 对 `requires_scopes=True` 的凭据类型，补充调用 `with_scopes()`
4. 对 GCE 凭据（无法 refresh 的），直接返回 `None` 并提示用户重新认证

**关键细节：**
- 源码级抑制了 ADC 用户凭据的 "no quota project" UserWarning（因为每个请求都带有 `X-Goog-User-Project: 1014160490159`，指向 Colab 的项目 ID）
- 若 scope 不足（缺少 `colaboratory`），keep-alive 会返回 403 `SCOPE_NOT_PERMITTED`，CLI 在 `colab new` 时就会做 pre-flight 检测并给出修复指引

### 2.4 认证排错

```bash
# 调试当前凭据身份、scope、过期时间
colab whoami

# 查看 token 信息
cat ~/.config/colab-cli/token.json | python -m json.tool

# ADC 登录（完整 scope）
gcloud auth application-default login \
    --scopes=openid,https://www.googleapis.com/auth/cloud-platform,\
https://www.googleapis.com/auth/userinfo.email,\
https://www.googleapis.com/auth/colaboratory
```

---

## 3. 会话生命周期

### 3.1 会话创建（`colab new`）

**源码路径：** `commands/session.py:new()`

```
1. 生成 session name（用户指定 或 uuid4 前 6 位）
2. 解析 accelerator：--gpu T4/L4/G4/A100/H100 或 --tpu v5e1/v6e1
3. Client.assign(uuid4(), variant, accelerator)
   ├── GET  /tun/m/assign?nbh=<uuid>&variant=<GPU|TPU>&accelerator=<T4|A100|...>
   │   └── 返回 xsrf_token + 已有 assignment（若存在）
   └── POST /tun/m/assign?nbh=<uuid>...  + X-Goog-Colab-Token header
       └── 返回 endpoint + runtime_proxy_info (token, url)
4. 创建 SessionState 对象
5. Pre-flight keep-alive: 发送 KeepAliveAssignment RPC
   ├── 403 + SCOPE_NOT_PERMITTED → 立即 unassign + 提示 scope 不足
   └── 其他错误 → 不阻塞，daemon 会重试
6. 持久化 SessionState → sessions.json
7. spawn_keep_alive() → 启动守护进程子进程
8. 更新 sessions.json（记录 keep_alive_pid）
9. 记录 session_created 事件到 history
```

**加速器映射（源码：** `session.py:_hardware_label()` 和 `new()` 中的 mapping）：

```python
# GPU
{"a100": A100, "h100": H100, "l4": L4, "t4": T4, "g4": G4}
# 未识别值 → 静默 fallback 到 A100，然后后端返回 400

# TPU
{"v5e1": V5E1, "v6e1": V6E1}
# 其他值 → fallback 到 V6E1

# CPU（无 --gpu/--tpu）
Accelerator.NONE, Variant.DEFAULT
```

**陷阱：** 未识别的 `--gpu` 值不会报错，而是静默 fallback 到 A100。若 A100 也无配额，后端返回 400，CLI 提示 "Backend rejected accelerator"。

### 3.2 会话状态同步（`colab sessions` / `colab status`）

**源码路径：** `common.py:State.sync_sessions()`

```
1. 读取 local sessions.json
2. 调用 Client.list_assignments() → GET /tun/m/assignments
3. 匹配 local (by endpoint) ↔ server-side
4. 修剪：local 中存在但 server 不存在的 → prune_session()
   ├── kill_process(keep_alive_pid)
   └── 从 sessions.json 删除
5. server 中存在但 local 不存在的 → 显示为 [?]
```

**设计意图：** 首次 `sync_sessions()` 结果被缓存到 `self._sessions`，后续同进程内的调用直接返回缓存。

### 3.3 会话停止（`colab stop`）

**源码路径：** `commands/session.py:stop()`

```
1. kill_process(keep_alive_pid) → SIGTERM × 5 次重试（每次 100ms 间隔）
2. ColabRuntime.stop(shutdown_kernel=True) → 关闭 WebSocket + 关闭 channels
3. Client.unassign(endpoint)
   ├── GET  /tun/m/unassign/<endpoint> → 获取 xsrf_token
   └── POST /tun/m/unassign/<endpoint> + X-Goog-Colab-Token
4. StateStore.remove(name) → 从 sessions.json 删除
5. 记录 session_terminated 事件
```

### 3.4 自动修剪

`colab sessions` 列表中将服务器端不存在的会话自动标记为 `[?]`（orphan）。当本地会话的 endpoint 不在服务器返回的 assignments 中时，会自动 prune。

---

## 4. Keep-Alive 守护进程

### 4.1 启动机制

**源码路径：** `commands/session.py:spawn_keep_alive()`

```python
# 构造命令
cmd = [
    sys.executable, "-m", "colab_cli.cli",
    f"--auth={auth_provider.value}",
    "--config", config_path,        # 传递 --config，确保 daemon 读写同一个 sessions.json
    "keep-alive", endpoint, session_name
]

# 分离进程
subprocess.Popen(
    cmd,
    start_new_session=True,          # POSIX: 脱离父进程会话
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    stdin=subprocess.DEVNULL,
)
```

**关键设计：**
- 使用 `start_new_session=True`（等同 `setsid`）使子进程脱离父进程的进程组，父进程退出后子进程不会被 SIGHUP
- 传递 `--auth` 和 `--config` 全局标志，确保 daemon 使用与父进程相同的认证方式和状态文件
- daemon PID 被记录在 `SessionState.keep_alive_pid` 中

### 4.2 Keep-Alive 循环

**源码路径：** `commands/session.py:keep_alive()`

```
loop (最长 24 小时):
  1. 检查 sessions.json 中该 session 是否仍存在
     ├── 不存在 → reason=session_not_found → 退出
     └── endpoint 不匹配 → reason=endpoint_mismatch → 退出
  2. Client.keep_alive_assignment(endpoint)
     → POST https://colab.pa.googleapis.com/$rpc/.../KeepAliveAssignment
  3. 错误处理:
     ├── 4xx 错误 → consecutive_4xx += 1
     │   └── 连续 2 次 4xx → reason=consecutive_4xx_errors → 退出
     └── 其他错误（网络等）→ 记录但不计数，继续重试
  4. sleep(60 秒)
```

**keep_alive_assignment RPC 细节（源码：** `client.py:keep_alive_assignment()`）：

```python
# HTTP POST 到 gRPC-Web 端点
POST https://colab.pa.googleapis.com/$rpc/google.internal.colab.v1.RuntimeService/KeepAliveAssignment
Headers:
  Content-Type: application/json+protobuf
  X-Goog-Api-Key: <public API key>        # 从二进制注册表解包
  x-user-agent: grpc-web-javascript/0.1    # 模拟前端
  x-goog-api-client: grpc-web/0.1          # 必需，否则后端返回 400
  X-Goog-User-Project: 1014160490159       # Colab 项目 ID
Body: [endpoint]                           # JSON 数组
```

**注意：** 公网 API Key 是通过二进制注册表（`_PUBLIC_CLIENT_REGISTRY`）以混淆方式存储的，调用 `_registry_field(0)` 解码 header 名，`_registry_field(1)` 解码 key 值。

### 4.3 Daemon 退出条件

| 退出原因 | 触发条件 |
|---------|---------|
| `time_limit_reached` | 运行超过 24 小时 |
| `session_not_found` | 本地 sessions.json 中该 session 被删除 |
| `endpoint_mismatch` | session 的 endpoint 已变化 |
| `consecutive_4xx_errors` | 连续 2 次 4xx 响应 |

每次退出都会写入 `keep_alive_stopped` 事件到 history。

---

## 5. 代码执行机制

### 5.1 ColabRuntime 内部结构

**源码路径：** `runtime.py`

```python
class ColabRuntime:
    def __init__(self, url, token, kernel_id=None, session_id=None, ...):
        # WebSocket 连接到远程 Jupyter Kernel
        # 使用 jupyter-kernel-client 库

    @property
    def kernel_client(self):
        # 惰性初始化 + 重试逻辑
        # retries=3, backoff=2 (1s, 2s, 4s)
        # extra_params: {"colab-runtime-proxy-token": self.token}
        # Headers: X-Colab-Client-Agent, X-Colab-Runtime-Proxy-Token
        # _own_kernel = False  ← 防止自动删除远程 kernel
```

### 5.2 `colab exec` 执行流程

**源码路径：** `commands/execution.py:exec_command()`

```
1. 解析 session（自动选择唯一 session 或使用 -s 指定）
2. 创建 ColabRuntime（惰性连接 kernel）
3. 前置步骤: os.chdir('/content')  # 确保在标准工作目录
4. 对每个 code block:
   ├── 更新 SessionState.running + last_execution → store.add()
   ├── runtime.execute_code(code, output_hook=..., timeout=...)
   │   ├── 流式输出: output_hook 逐条推送
   │   └── 完整返回: outputs list
   └── 记录 execution 事件到 history
5. 对 .ipynb 文件：保存 _output.ipynb（带输出 cell）
6. runtime.stop() → 关闭 channels 但不 shutdown kernel
```

### 5.3 输入源处理

| 输入源 | 检测方式 | 行为 |
|--------|---------|------|
| `-f file.py` | 文件扩展名 | 读取整个文件作为单个 code block |
| `-f file.ipynb` | `.ipynb` 后缀 | 逐个执行 code cell，保存输出 |
| 管道 stdin（非 TTY）| `sys.stdin.isatty()` | 读取 stdin 全部内容执行 |
| 无输入（TTY）| `is_stdin_tty()` | 报错退出 |

### 5.4 Timeout 机制

`execute_code` 的 `timeout` 参数最终传递给 `jupyter_kernel_client` 的 poll 循环。默认 `timeout=10` 秒。

**注意：** 这不是总执行时间限制，而是 kernel 连续无输出的最大等待时间。长时间运行的训练任务只要持续产生输出（print、log），就不会触发 timeout。

### 5.5 输出渲染

- `stream` 输出（stdout/stderr）→ 直接写入终端
- `text/plain` → `typer.echo()`
- `image/png` / `image/jpeg` → Kitty 终端协议内联渲染 + 保存到临时文件
- `error` → traceback 写入 stderr

### 5.6 `colab console`（原始 TTY）

**源码路径：** `console.py`

- 通过 WebSocket 连接 `/colab/tty?colab-runtime-proxy-token=<token>`
- 设置终端为 raw 模式（`tty.setraw()`）
- 将本地 stdin 逐字符转发到远程 shell
- 支持 SIGWINCH 信号 → 实时同步终端尺寸
- 非 TTY 输入（管道）时：EOF 后发送 `exit\n` + 等待 0.5 秒再关闭

---

## 6. 文件传输协议

### 6.1 Contents API

**源码路径：** `contents.py`

所有文件操作通过 Jupyter Contents API 进行：

```
{base_url}/api/contents/{path}?colab-runtime-proxy-token={token}&authuser=0
```

### 6.2 上传（`colab upload`）

```python
# Base64 编码后 PUT
PUT /api/contents/{remote_path}
Body: {
    "name": filename,
    "path": remote_path,
    "type": "file",
    "format": "base64",
    "content": base64.b64encode(file_content),
    "chunk": 1
}
```

### 6.3 下载（`colab download`）

```
GET /api/contents/{remote_path}?content=1
→ 返回 JSON: {content: "<base64>", format: "base64", type: "file"}
→ 解码 base64 → 写入本地文件
```

目录无法下载（抛 `IsADirectoryError`）。

### 6.4 `colab edit`

**源码路径：** `commands/files.py:edit()`

```
1. 下载远程文件到 NamedTemporaryFile（如无则创建空文件）
2. 计算 SHA256 hash
3. 调用 click.edit() → 打开 $EDITOR
4. 再次计算 hash
5. 若有变更 → 上传覆盖
```

### 6.5 文件操作命令

| 命令 | HTTP 方法 | 端点 |
|------|----------|------|
| `ls` | GET | `/api/contents/{path}` |
| `upload` | PUT | `/api/contents/{path}` |
| `download` | GET | `/api/contents/{path}?content=1` |
| `rm` | DELETE | `/api/contents/{path}` |
| `edit` | GET + PUT | 下载 → 编辑 → 上传 |

---

## 7. `colab run`：一次性任务引擎

### 7.1 设计理念

`colab run` 将 `new → exec → stop` 三合一，支持 shebang 直接执行：

```python
#!/usr/bin/env -S colab run --gpu T4 --keep
import torch
print(torch.cuda.get_device_name(0))
```

### 7.2 脚本包装机制

**源码路径：** `commands/run.py:_build_script_payload()`

```python
payload = """
import sys, warnings
sys.argv = ['script.py', 'arg1', 'arg2']     # 模拟原生 python 调用
__name__ = '__main__'                         # 确保 if __name__ == '__main__' 生效
warnings.filterwarnings('ignore', message="To exit: use")  # 抑制 IPython 退出警告
# --- 用户脚本内容（去除 shebang）---
"""
```

### 7.3 SystemExit 处理

```python
# _exit_code_from_outputs():
# - SystemExit(None/0) → exit 0（静默，不打印 traceback）
# - SystemExit(N)     → exit N
# - SystemExit('msg') → exit 1
# - 其他 Exception    → exit 1
```

### 7.4 清理流程

```
finally:
  if not keep:
    _teardown(name, s, reason)
    ├── kill_process(keep_alive_pid)
    ├── ColabRuntime.stop(shutdown_kernel=True)
    ├── client.unassign(endpoint)
    ├── store.remove(name)
    └── 记录 session_terminated 事件
```

### 7.5 `--keep` 选项

使用 `--keep` 后，VM 在脚本执行完成后**不会被释放**，用户可以：
- 通过 `colab status -s <name>` 查看状态
- 通过 `colab exec -s <name>` 继续执行代码
- 手动 `colab stop -s <name>` 释放

---

## 8. 自动化命令内部机制

### 8.1 `colab install`

**源码路径：** `commands/automation.py:install()`

```python
# 在远程 kernel 中执行：
# 首选 uv（Colab VM 预装 uv）
subprocess.check_call(['uv', 'pip', 'install', '--system'] + packages)
# 失败时 fallback 到 pip
subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + packages)
```

`-r requirements.txt` 会先上传 requirements 文件到 `/content/`，再执行 `uv pip install --system -r /content/requirements.txt`。

### 8.2 `colab drivemount`

**源码路径：** `commands/automation.py:drivemount()`

```
1. 在 kernel 中执行: from google.colab import drive; drive.mount(path)
2. 拦截 colab_request 消息（WebSocket hook）
3. 若 authType == "dfs_ephemeral":
   ├── 通过 credential propagation API 自动授权
   │   GET/POST /tun/m/credentials-propagation/<endpoint>
   │   参数: authtype=dfs_ephemeral, version=2
   └── 若需手动授权 → 打印浏览器 URL → 等待用户按 Enter
4. 将 colab_reply 消息发回 kernel，恢复 mount 流程
```

**Timeout 特殊处理：** 使用 `INTERACTIVE_AUTOMATION_TIMEOUT_SEC = 600`（10 分钟），因为用户在浏览器中完成 OAuth 流程可能需要时间。

### 8.3 `colab auth`

在 kernel 中执行 `google.colab.auth.authenticate_user()`，同样使用 600 秒 timeout。

### 8.4 WebSocket Hook 机制

**源码路径：** `runtime.py:_apply_ws_hook()`

monkey-patch WebSocket 的 `on_message` 处理函数，拦截 `msg_type == "colab_request"` 的消息。若 hook 返回 `True`，消息不会传递到原始处理器。

这实现了 Drive 挂载过程中的认证拦截和凭据传播。

---

## 9. 状态持久化与锁机制

### 9.1 文件结构

```
~/.config/colab-cli/
├── sessions.json       # 会话状态（StateStore）
├── settings.json       # 全局设置（SettingsStore）
├── token.json          # OAuth2 token 缓存
├── colab.log           # 调试日志
└── history/
    ├── <session1>.jsonl
    └── <session2>.jsonl
```

### 9.2 文件锁

**源码路径：** `state.py:_LockedFileStore`

使用 `fcntl.flock()` 实现 POSIX 文件锁：

- `_lock_shared()` → `LOCK_SH`：读操作（`get()`, `list()`）
- `_lock_exclusive()` → `LOCK_EX`：写操作（`add()`, `remove()`）

**这保证了多进程并发安全：** keep-alive daemon 和用户 CLI 命令可能同时读写 `sessions.json`。

### 9.3 SessionState 数据模型

```python
class SessionState(BaseModel):
    name: str                          # 会话名称
    token: str                         # runtime proxy token
    url: str                           # runtime proxy URL
    endpoint: str                      # assignment endpoint
    variant: str = "DEFAULT"           # DEFAULT / GPU / TPU
    accelerator: str = "NONE"          # T4 / A100 / V5E1 / ...
    kernel_id: Optional[str]           # Jupyter kernel ID
    session_id: Optional[str]          # Jupyter session ID
    last_execution: Optional[Tuple[str, Optional[str], str]]  # (文件, cell_id, 时间)
    running: Optional[str]             # 当前运行描述 (None = IDLE)
    keep_alive_pid: Optional[int]      # keep-alive 进程 PID
```

### 9.4 History 事件类型

| event_type | 触发时机 |
|-----------|---------|
| `session_created` | `colab new` / `colab run` |
| `session_terminated` | `colab stop` / run 结束 / prune |
| `execution` | 每次 `colab exec` / `colab run` / REPL 执行 |
| `file_operation` | `ls`, `rm`, `upload`, `download`, `edit` |
| `automation` / `automation_result` | `auth`, `install`, `drivemount` |
| `stdin_request` / `input_reply` | 交互式输入 |
| `keep_alive_started` / `keep_alive_stopped` | daemon 生命周期 |
| `keep_alive_error` | daemon 遇到错误 |
| `repl_started` / `console_started` | REPL / console 会话 |
| `drive_auth_needed` / `drive_auth_success` | Drive 挂载认证 |
| `colab_request` | WebSocket colab_request 拦截 |

### 9.5 导出格式

`colab log -s <name> -o <file>` 支持四种输出格式（由文件后缀决定）：

| 后缀 | 格式 |
|------|------|
| `.ipynb` | Jupyter Notebook（cell 级重建） |
| `.md` | Markdown（代码块 + 输出） |
| `.jsonl` | 原始 JSONL（逐行 JSON） |
| `.txt` | 纯文本 |

---

## 10. 更新检测系统

### 10.1 两级策略

**源码路径：** `auto_update.py` + `cli.py:callback()`

```
每次运行 colab <command>:
  1. 抑制列表检查（不检查 update/version/log/pay/help/url/whoami/readme/skill）
  2. 从 settings.json 读取 last_check
  3. 若距上次检查 ≥ 1 天:
     ├── fetch PyPI JSON (https://pypi.org/pypi/google-colab-cli/json)
     ├── 比较 version
     ├── 若有新版本: 打印 banner + 更新 cache
     └── 更新 last_check + latest_version → settings.json
  4. 若不足 1 天:
     └── 从 cache 中读取 latest_version，若有新版本则显示 banner（含 cached 标记）
```

### 10.2 自升级

```bash
colab update --install
```

检测是通过 `uv tool install` 还是 `pip install` 安装，并使用相应命令升级。

---

## 11. 实战陷阱与排错手册

### 11.1 后台训练 stdout 无输出

**原因：** Python 子进程 stdout 默认全缓冲。

**解决：** 在 launch script 中设置：
```python
import subprocess
proc = subprocess.Popen(
    ["python", "-u", "train.py"],  # -u 禁用缓冲
    env={"PYTHONUNBUFFERED": "1", **os.environ},
    stdout=open("/content/train.log", "w"),
    stderr=subprocess.STDOUT,
    start_new_session=True,  # 防止 SIGHUP
)
```

### 11.2 `colab exec -f` 只能用相对路径

```
❌ colab exec -f /content/train.py    # FileNotFoundError
✅ colab exec -f train.py             # 正确
```

**原因：** Typer 的 `-f` 选项在本地读取文件，然后发送内容到远程执行。路径是本地路径。

### 11.3 Free Tier VM 自动终止

Free tier VM 通常在 **2-4 小时**后自动终止。keep-alive 机制只能防止空闲回收，不能绕过 Google 的硬时限。必须靠 checkpoint + download 实现持久化。

### 11.4 未识别 GPU fallback 到 A100

```
colab new --gpu V100   # 静默 fallback 到 A100 → 然后后端返回 400
```

始终使用官方支持的加速器值：`T4`, `L4`, `G4`, `A100`, `H100`, `v5e1`, `v6e1`。

### 11.5 `colab sessions` 显示 `[?]`

表示服务器上存在一个 assignment，但本地 `sessions.json` 中没有对应记录。这通常是以下情况导致的：
- 在另一台机器上创建了 session
- 手动删除了 `sessions.json`
- 使用非默认 `--config` 路径

用 `colab stop -s <endpoint>` 无法停止，因为本地没有这个 session。需要通过浏览器访问 `https://colab.research.google.com` 手动关闭。

### 11.6 OAuth Scope 不足

**症状：** `colab new` 后 session 很快消失，或 `colab status` 显示 session 丢失。

**检查：**
```bash
colab whoami | grep Scopes -A 20
# 必须包含: https://www.googleapis.com/auth/colaboratory
```

**修复：**
- OAuth2: 删除 `~/.config/colab-cli/token.json`，重新运行 `colab new`
- ADC: 重新执行 `gcloud auth application-default login --scopes=...,https://www.googleapis.com/auth/colaboratory`

### 11.7 Kernel 连接超时

ColabRuntime 初始化时会重试 3 次（指数退避 1s, 2s, 4s）。若 3 次都失败，说明 VM 可能尚未就绪或网络不稳定。

### 11.8 交互式命令在 Agent/脚本中挂死

`colab repl` 和 `colab console` 期望 TTY。在非 TTY 环境中：
- `repl`：管道输入会自动退化为非交互模式
- `console`：建议用 `echo "cmd" | colab console` 替代交互式使用

### 11.9 多会话自动选择

当只有一个活跃 session 时，可省略 `-s`。当有多个时，**必须**指定 `-s`，否则 CLI 报错退出。

### 11.10 代理配置

如需要代理访问 Colab API，可设置环境变量：
```bash
HTTPS_PROXY=http://127.0.0.1:7890 colab new
```

---

## 12. 命令行完整参考

### 全局标志

| 标志 | 默认值 | 说明 |
|------|--------|------|
| `--auth {oauth2,adc}` | `oauth2` | 认证策略 |
| `-c, --client-oauth-config PATH` | `~/.colab-cli-oauth-config.json` | OAuth 客户端配置 |
| `--config PATH` | `~/.config/colab-cli/sessions.json` | 会话状态文件路径 |
| `--logtostderr` | False | 调试日志输出到 stderr |

### 会话管理

```bash
colab new [-s NAME] [--gpu T4|L4|G4|A100|H100] [--tpu v5e1|v6e1]
colab sessions                                          # 列出所有活跃会话
colab status [-s NAME]                                  # 显示会话详情
colab restart-kernel [-s NAME]                          # 重启 kernel
colab stop [-s NAME]                                    # 停止会话并释放 VM
colab url [-s NAME] [--open] [--host URL]               # 生成浏览器连接 URL
```

### 代码执行

```bash
colab run [--gpu GPU] [--tpu TPU] [--keep] [--timeout N] SCRIPT [ARGS...]
colab exec [-s NAME] [-f FILE] [--output-image PATH] [--timeout N]
colab repl [-s NAME] [--output-image PATH]
colab console [-s NAME]
```

### 文件操作

```bash
colab ls [-s NAME] [PATH]
colab upload [-s NAME] LOCAL REMOTE
colab download [-s NAME] REMOTE LOCAL
colab rm [-s NAME] PATH
colab edit [-s NAME] PATH
```

### 自动化工具

```bash
colab auth [-s NAME]                                    # GCP 认证
colab drivemount [-s NAME] [PATH]                       # 挂载 Google Drive
colab install [-s NAME] [-r FILE] [PKG...]              # 安装包
```

### 日志与状态

```bash
colab log [-s NAME] [-n LINES] [-t TYPE] [-o OUTPUT]    # 查看/导出日志
colab pay                                                 # 打开订阅管理页面
colab version                                             # 显示版本
colab update [--install]                                  # 检查/执行更新
colab whoami                                              # 调试：查看当前凭据
colab help [COMMAND]                                      # 帮助
```

---

## 附录 A：环境变量

| 变量 | 作用 |
|------|------|
| `GOOGLE_APPLICATION_CREDENTIALS` | ADC 凭据文件路径 |
| `HTTPS_PROXY` / `HTTP_PROXY` | 代理设置 |
| `PYTHONUNBUFFERED=1` | 禁用 Python 输出缓冲 |
| `EDITOR` | `colab edit` 使用的编辑器 |

## 附录 B：相关文件路径

| 路径 | 内容 |
|------|------|
| `~/.config/colab-cli/sessions.json` | 会话状态（JSON） |
| `~/.config/colab-cli/settings.json` | 全局设置 |
| `~/.config/colab-cli/token.json` | OAuth2 token 缓存 |
| `~/.config/colab-cli/colab.log` | 调试日志 |
| `~/.config/colab-cli/history/*.jsonl` | 会话事件日志 |
| `~/.colab-cli-oauth-config.json` | OAuth 客户端配置 |

---

> 本文档基于 `google-colab-cli v0.5.9` 源码分析编写。CLI 在持续更新中，部分内部实现细节可能随版本变化。
