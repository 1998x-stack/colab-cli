# google-colab-cli: Source Code Architecture

**Version:** 0.5.11 | **License:** Apache 2.0 (Google LLC) | **Language:** Python 3.13  
**Install path:** `~/.local/share/uv/tools/google-colab-cli/lib/python3.13/site-packages/colab_cli/`  
**Entry point:** `colab_cli.cli:main()` → Typer CLI app

---

## 1. Module Map

```
colab_cli/
├── cli.py              ← Entry point, Typer app, global callback, subcommand registration
├── client.py           ← REST API client, Colab backend communication, KeepAliveAssignment RPC
├── runtime.py          ← WebSocket lifecycle, kernel execution, output hooks, WS message interception
├── common.py           ← Global State singleton, session sync, process management
├── state.py            ← SessionState/SettingsStore/StateStore persistence with file locking
├── auth.py             ← OAuth2 InstalledAppFlow + ADC credential management
├── history.py          ← JSONL-structured session event logger
├── contents.py         ← File I/O via Jupyter Contents API (upload/download/ls/rm)
├── console.py          ← Raw TTY terminal via WebSocket (colab console)
├── repl.py             ← Interactive Python REPL via prompt_toolkit (colab repl)
├── converter.py        ← History export: .ipynb, .md, .txt, .jsonl
├── utils.py            ← Status code extraction, image rendering (Kitty protocol), terminal error detection
├── auto_update.py      ← Background PyPI version check, self-install
├── oauth_config.json   ← Bundled OAuth2 client config (fallback)
├── COLAB_SKILL.md      ← Bundled skill documentation
├── README.md           ← Bundled readme
└── commands/
    ├── session.py      ← new, stop, status, sessions, restart-kernel, keep-alive daemon
    ├── execution.py    ← exec, repl, console (code execution entry points)
    ├── files.py        ← ls, rm, upload, download, edit
    ├── run.py          ← colab run: one-shot provision→execute→teardown
    ├── automation.py   ← auth (gcloud), drivemount, install (pip/uv)
    └── utility.py      ← pay, url, log, version, update, whoami, readme, skill
```

---

## 2. Architecture Layers

```
┌─────────────────────────────────────────────────┐
│  CLI Layer (cli.py + commands/*.py)              │
│  Typer commands, argument parsing, output format │
├─────────────────────────────────────────────────┤
│  Orchestration Layer (common.py State singleton) │
│  Session lifecycle, credential caching,          │
│  multi-account isolation via $HOME               │
├─────────────────────────────────────────────────┤
│  Transport Layer (client.py + runtime.py)        │
│  ┌──────────────┐  ┌──────────────────────┐     │
│  │ REST client   │  │ WebSocket runtime     │     │
│  │ (requests)     │  │ (jupyter-kernel-client)│    │
│  │ colab.pa.     │  │ *.prod.colab.dev      │     │
│  │ googleapis.com│  │ (Jupyter kernel proxy) │     │
│  └──────────────┘  └──────────────────────┘     │
├─────────────────────────────────────────────────┤
│  Storage Layer (state.py + history.py)           │
│  File-locked JSON persistence, JSONL event log   │
├─────────────────────────────────────────────────┤
│  Auth Layer (auth.py)                            │
│  OAuth2 InstalledAppFlow + ADC credential flow   │
└─────────────────────────────────────────────────┘
```

---

## 3. Entry Point & CLI Structure

### 3.1 `cli.py` — Application Bootstrap

```python
app = typer.Typer(no_args_is_help=True, cls=AlphabeticalGroup)

@app.callback()
def callback(ctx, client_oauth_config, config, logtostderr, auth):
    state.client_oauth_config = client_oauth_config
    state.config_path = config            # session state file path
    state.logtostderr = logtostderr       # debug logging toggle
    state.auth_provider = auth            # oauth2 (default) or adc
    setup_logging(logtostderr)
    auto_update.run_background_check()    # daily PyPI version check
```

**Global flags** (`--auth`, `--config`, `--client-oauth-config`, `--logtostderr`)  
are set once at the callback level and stored in the `State` singleton.  
All subcommands access them via `from colab_cli.common import state`.

**Subcommand registration** is grouped by functional domain:
```python
session.register(app)     # new, stop, status, sessions, restart-kernel, keep-alive
execution.register(app)   # exec, repl, console
files.register(app)       # ls, rm, upload, download, edit
automation.register(app)  # auth, drivemount, install
run.register(app)         # run (one-shot)
utility.register(app)     # pay, url, log, version, update, whoami, readme, skill
```

`AlphabeticalGroup` sorts subcommands alphabetically in `--help` output regardless  
of registration order — user-facing discoverability over internal grouping.

### 3.2 `common.py` — Global State Singleton

```python
class State:
    client_oauth_config: str    # ~/.colab-cli-oauth-config.json
    config_path: Optional[str]  # ~/.config/colab-cli/sessions.json
    logtostderr: bool
    auth_provider: AuthProvider  # OAUTH2 or ADC

    @property
    def client(self) -> Client:      # Lazy: creates Client(Prod(), credentials)
    @property
    def store(self) -> StateStore:   # Lazy: StateStore(config_path)
    @property
    def history(self) -> HistoryLogger:  # Lazy: HistoryLogger()

    def sync_sessions(self):         # Cross-reference local state with server assignments
    def resolve_session(self, name): # Pick unique session or error
    def prune_session(self, name):   # Kill keep-alive, remove from store
```

`state = State()` is a module-level singleton. Every command imports it.  
The lazy properties mean credentials aren't minted until the first API call.

---

## 4. REST API Client (`client.py`)

### 4.1 Two Backend Domains

```python
class Prod(ColabEnvironment):
    domain = "https://colab.research.google.com"    # Frontend REST API
    api   = "https://colab.pa.googleapis.com"       # gRPC-web RPC API
```

| Domain | Purpose | Auth | Endpoints |
|--------|---------|------|-----------|
| `colab.research.google.com` | Session management, file I/O | OAuth2 Bearer + cookie | `/tun/m/assignments`, `/tun/m/assign`, `/tun/m/unassign` |
| `colab.pa.googleapis.com` | Runtime keep-alive | OAuth2 + API key + project header | `/$rpc/google.internal.colab.v1.RuntimeService/KeepAliveAssignment` |

### 4.2 Session Assignment Flow

```
colab new --gpu T4
  → Client.assign(notebook_hash, variant=GPU, accelerator=T4)
    → GET  /tun/m/assign?nbh=...&variant=GPU&accelerator=T4
      ← GetAssignmentResponse { token: xsrf_token }
    → POST /tun/m/assign?nbh=...&variant=GPU&accelerator=T4
        (headers: X-Goog-Colab-Token: xsrf_token)
      ← PostAssignmentResponse { endpoint, runtime_proxy_info: {token, url} }
```

### 4.3 KeepAliveAssignment RPC (The Broken One)

```python
def keep_alive_assignment(self, endpoint: str):
    url = "https://colab.pa.googleapis.com/$rpc/.../KeepAliveAssignment"
    headers = {
        "Content-Type": "application/json+protobuf",
        "x-goog-api-key": "<Colab's public web-client API key>",
        "x-goog-user-project": "1014160490159",  # ← The problem
        "x-user-agent": "grpc-web-javascript/0.1",
        "x-goog-api-client": "grpc-web/0.1",
    }
    self._issue_request(url, method="POST", headers=headers, json=[endpoint])
```

**The IAM deadlock:** `x-goog-user-project: 1014160490159` tells Google to bill  
Colab's project. Google checks if the user has `serviceusage.services.use` on  
project 1014160490159 — they don't → **HTTP 403 USER_PROJECT_DENIED**.

Without the header → **HTTP 400 CONSUMER_INVALID** (API key project ≠ user project).

The API key is obfuscated in `_PUBLIC_CLIENT_REGISTRY`:
```python
_PUBLIC_CLIENT_REGISTRY = (
    b"\x1c" b"782d676f6f672d6170692d6b6579"   # length=28, hex="x-goog-api-key"
    b"\x4e" b"41497a61537941324276..."         # length=78, hex="AIzaSyA2Bv..."
)
```

### 4.4 Request Infrastructure

`_issue_request()` handles XSSI prefix stripping (`)]}'\n`), JSON parsing  
via Pydantic models, and error wrapping in `ColabRequestError`.  
The `requests.AuthorizedSession` from `google-auth` handles Bearer token injection.

---

## 5. WebSocket Runtime (`runtime.py`)

### 5.1 ColabRuntime

```python
class ColabRuntime:
    url: str              # https://8080-<endpoint>.prod.colab.dev
    token: str            # JWT runtime proxy token
    kernel_id: str        # Jupyter kernel UUID
    session_id: str       # Jupyter session UUID
    _kernel_client: KernelClient  # jupyter_kernel_client instance
```

### 5.2 Connection Lifecycle

```
ColabRuntime.kernel_client (lazy property)
  → jupyter_kernel_client.KernelClient(server_url, token, kernel_id)
    → KernelClient.start()
      → KernelHttpManager.start_kernel()     # HTTP POST to create kernel
      → KernelWebSocketClient.start_channels()
        → WebSocketApp(url, headers, subprotocols)
        → Thread → run_forever(ping_interval=60, reconnect=0)
        → Channels: shell, iopub, stdin, hb (heartbeat), control
```

**Connection retry:** 3 attempts with exponential backoff (2s, 4s) on  
`ReadTimeout`/`ConnectTimeout`. Other exceptions propagate immediately.

**Kernel lifecycle:** `_own_kernel = False` prevents the client from deleting  
the kernel on disconnect. The CLI manages kernel lifecycle explicitly.

### 5.3 WebSocket Hook System

`runtime.py:46-87` installs a `colab_request` hook on the kernel WebSocket's  
`on_message` handler. This intercepts Colab-specific messages (Drive auth  
requests) and routes them to `runtime.colab_request_hook` without passing  
through to the default message handler.

Used by `automation.py` for `drivemount` — intercepts the kernel's  
`dfs_ephemeral` auth request, completes the OAuth flow via REST,  
and sends the reply back on the stdin channel.

### 5.4 Code Execution

```python
def execute_code(self, code, allow_stdin=False, output_hook=None, timeout=None):
    if output_hook:
        # Streaming mode: execute_interactive with per-message output hook
        reply = self.kernel_client.execute_interactive(code, output_hook=wrapped_hook)
    else:
        # Batch mode: execute, return all outputs at once
        reply = self.kernel_client.execute(code)
```

Default timeout is `REQUEST_TIMEOUT` (10s) from jupyter_kernel_client.  
The `--timeout` flag on `colab exec` overrides this.

### 5.5 WebSocket Stop Sequence

```python
def stop(self, shutdown_kernel=False):
    client.stop_channels()       # Stop all Jupyter channels
    client.kernel_socket.close() # Close WebSocket
    if shutdown_kernel:
        manager.shutdown_kernel(now=True)  # HTTP DELETE kernel
```

---

## 6. Session Lifecycle (`commands/session.py`)

### 6.1 Session Creation

```
colab new --gpu T4 -s <name>
  → Client.assign(uuid4(), variant=GPU, accelerator=T4)
  → Pre-flight: Client.keep_alive_assignment(endpoint)
      Catches: SCOPE_NOT_PERMITTED → unassign + remediation message + exit
      Misses:  USER_PROJECT_DENIED → passes through silently (bug)
  → StateStore.add(SessionState)
  → spawn_keep_alive(endpoint, name) → subprocess.Popen(daemon)
  → StateStore.add(updated)  # now includes keep_alive_pid
```

### 6.2 Keep-Alive Daemon

Spawned as a **detached subprocess** via `colab_cli.cli keep-alive <endpoint> <name>`.  
Uses `start_new_session=True` (POSIX) or `DETACHED_PROCESS` (Windows).

**Daemon loop** (`keep_alive()` function):
```python
while time.time() - start_time < 24 * 3600:  # 24h max
    s = state.store.get(session_name)
    if not s: break                           # Session removed locally
    if s.endpoint != endpoint: break          # Endpoint changed

    try:
        state.client.keep_alive_assignment(endpoint)
        consecutive_4xx = 0
    except Exception as e:
        if 400 <= status_code < 500:
            consecutive_4xx += 1
            if consecutive_4xx >= 2: break    # 2 consecutive 4xx → exit

    time.sleep(60)                            # Every 60 seconds
```

**Exit reasons:**
- `time_limit_reached` — 24 hours elapsed
- `session_not_found` — session removed from local state
- `endpoint_mismatch` — endpoint changed (reassignment)
- `consecutive_4xx_errors` — 2 consecutive 4xx (the 403 USER_PROJECT_DENIED pattern)

### 6.3 Session Termination

```
colab stop -s <name>
  → kill_process(keep_alive_pid)        # SIGTERM daemon
  → ColabRuntime.stop(shutdown_kernel=True)
  → Client.unassign(endpoint)           # POST /tun/m/unassign/<endpoint>
  → StateStore.remove(name)
  → HistoryLogger.log_event(session_terminated)
```

---

## 7. Authentication (`auth.py`)

### 7.1 Two Providers

| Provider | Flag | Flow | Token Storage |
|----------|------|------|--------------|
| OAuth2 | `--auth=oauth2` (default) | InstalledAppFlow → local server on port 8200 | `~/.config/colab-cli/token.json` |
| ADC | `--auth=adc` | `google.auth.default()` → gcloud ADC | gcloud credential store |

### 7.2 OAuth2 Scopes

```python
PUBLIC_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/colaboratory",    # Required for KeepAliveAssignment
    "https://www.googleapis.com/auth/drive.file",       # Required for drivemount
]
```

Token refresh is automatic via `google-auth`'s `Credential.refresh()`.  
If refresh fails, the full OAuth2 flow re-runs (browser-based consent).

---

## 8. Data Flow: `colab exec`

```
User: colab exec -s mysession -f train.py --timeout 120

1. CLI (execution.py:exec_command)
   → state.resolve_session("mysession")
   → state.store.get("mysession") → SessionState
   → Read local file train.py
   → If .ipynb: parse with nbformat, extract code cells

2. Runtime setup (execution.py:160-167)
   → ColabRuntime(url, token, kernel_id, session_id)
   → runtime.execute_code("import os; os.chdir('/content')")

3. Code execution (runtime.py:execute_code)
   → kernel_client.execute_interactive(code, output_hook=display_output)
   → WebSocket: kernel_socket.send(execute_request)
   → WebSocket: kernel_socket.recv() → IOPub messages
   → output_hook called per message → display_output() prints to terminal

4. Cleanup (execution.py:228-231)
   → s.running = None; state.store.add(s)
   → runtime.stop()  # Close WebSocket channels + socket
   → If .ipynb: save output notebook
```

**Key constraint:** The kernel executes code **serially**. Only one `colab exec`  
can actively execute at a time. Additional execs have their code queued.

---

## 9. File I/O (`contents.py`)

`ContentsClient` wraps the Jupyter Contents API at  
`https://8080-<endpoint>.prod.colab.dev/api/contents/<path>`.

All requests carry `colab-runtime-proxy-token` as a query parameter  
for authentication (the same JWT from session creation).

| Operation | HTTP | Payload | Notes |
|-----------|------|---------|-------|
| `list_dir` | GET | — | Returns `{type, content: [...]}` |
| `upload` | PUT | `{name, path, type:"file", format:"base64", content: b64}` | Always base64-encoded |
| `download` | GET + `?content=1` | — | `content` field is base64-decoded |
| `rm` | DELETE | — | No JSON response |

**Directory download limitation:** The API returns `type: "directory"` for  
directories; attempting to download raises `IsADirectoryError`.  
Must tar on VM first, then download the tar.

---

## 10. State Persistence (`state.py`)

### 10.1 SessionState

```python
class SessionState(BaseModel):
    name: str
    token: str                # JWT runtime proxy token
    url: str                  # https://8080-<endpoint>.prod.colab.dev
    endpoint: str             # gpu-t4-s-kkb-...
    variant: str              # DEFAULT / GPU / TPU
    accelerator: str          # NONE / T4 / L4 / A100 / ...
    kernel_id: Optional[str]
    session_id: Optional[str]
    last_execution: Optional[Tuple[str, Optional[str], str]]
    running: Optional[str]    # Current operation description
    keep_alive_pid: Optional[int]
```

### 10.2 StateStore

File-locked JSON at `~/.config/colab-cli/sessions.json` (configurable via `--config`).  

Uses `filelock.ReadWriteLock` for cross-platform, multi-process-safe reads/writes:
- **Shared lock** for reads (multiple readers concurrently)
- **Exclusive lock** for writes (single writer, blocks all readers)
- `is_singleton=False` prevents reentrant lock conflicts across StateStore instances

### 10.3 SettingsStore

Same locking pattern at `~/.config/colab-cli/settings.json`.  
Stores update check metadata: `last_check`, `latest_version`, `enable_update_check`.

### 10.4 Multi-Account Isolation

Each account has a separate `$HOME`:
```
~/.config/colab-cli/              ← hackxie1998 (default, $HOME=~)
~/colab-accounts/account-b/.config/colab-cli/  ← stefaniehu929
~/colab-accounts/account-c/.config/colab-cli/  ← xbetterdetermine
~/colab-accounts/account-clb/.config/colab-cli/ ← xieminghack
```

Since all state paths (`sessions.json`, `token.json`, `settings.json`, `history/`)  
derive from `$HOME`, switching `$HOME` fully isolates accounts.

---

## 11. History & Export (`history.py`, `converter.py`)

### 11.1 HistoryLogger

JSONL files at `~/.config/colab-cli/history/<session_name>.jsonl`.  
One JSON object per line, append-only.

Event types:
- `session_created` / `session_terminated`
- `execution` (code + outputs + cell metadata)
- `keep_alive_started` / `keep_alive_error` / `keep_alive_stopped`
- `file_operation` (ls, rm, upload, download, edit)
- `automation` / `automation_result` (auth, install, drivemount)
- `stdin_request` / `input_reply`
- `colab_request` (Drive auth interception)
- `repl_started` / `console_started`

### 11.2 Converter

Exports history to multiple formats:
- `.ipynb` — Jupyter notebook with code cells and outputs
- `.md` — Markdown with code blocks
- `.txt` — Plain text timeline
- `.jsonl` — Raw JSONL copy

---

## 12. `colab run` — One-Shot Execution (`commands/run.py`)

Combines `new` + `exec` + `stop` into a single command. Designed for shebangs:

```python
#!/usr/bin/env -S colab run --gpu T4
import torch
print(torch.cuda.get_device_name(0))
```

**Key behaviors:**
- Validates script path locally BEFORE allocating a VM (no wasted quota)
- Wraps script body with `sys.argv` setup, `__name__ = '__main__'`, and IPython warning suppression
- `sys.exit(N)` in the script maps to exit code N
- `--keep` flag prevents auto-teardown (session stays alive for later use)
- `--timeout` defaults to 30s (different from colab exec's 10s)

---

## 13. Console & REPL

### 13.1 `colab console` (`console.py`)

Raw TTY connection via WebSocket to `/colab/tty` endpoint.  
Uses `termios.tcsetattr(fd, TCSANOW)` for raw mode.  
Handles SIGWINCH for terminal resize propagation.  
Stdin reading runs in a daemon thread; EOF sends `exit\n` to the remote shell.

### 13.2 `colab repl` (`repl.py`)

Interactive Python REPL using `prompt_toolkit` with:
- `PygmentsLexer(PythonLexer)` for syntax highlighting
- Multi-line support (Enter submits, Esc+Enter inserts newline)
- `InMemoryHistory` for session-scoped command history
- Rich `Console` for styled output
- `/quit`, `quit()`, `exit()` to exit

---

## 14. Automation Commands (`commands/automation.py`)

### 14.1 `colab auth`

Runs `google.colab.auth.authenticate_user()` on the VM with `USE_AUTH_EPHEM=0`.  
Timeout: 600s (10 min) for browser-based OAuth flow.

### 14.2 `colab drivemount`

Runs `google.colab.drive.mount(path)` on the VM.  
The `drivefs_hook` in `runtime.py` intercepts the kernel's DFS ephemeral auth  
request, completes the OAuth flow via REST to `colab.research.google.com`,  
and sends the reply back on the stdin channel.  
Timeout: 600s.

### 14.3 `colab install`

Installs Python packages via `uv pip install --system` (preferred) or  
`pip install` (fallback). Accepts package names or `-r requirements.txt`.  
Requirements files are uploaded to `/content/` before installation.

---

## 15. Known Issues & Gotchas (Source-Level)

### 15.1 KeepAliveAssignment IAM Deadlock

**Files:** `client.py:298-324`, `session.py:200-220`

The `x-goog-user-project: 1014160490159` header causes `USER_PROJECT_DENIED`  
for all users. The pre-flight check in `session.py:206-220` only catches  
`SCOPE_NOT_PERMITTED`, not `USER_PROJECT_DENIED`. Result: every session's  
keep-alive daemon dies 61 seconds after creation.

### 15.2 WebSocket No Reconnect

**File:** `wsclient.py:1279-1281`

```python
self.kernel_socket.run_forever(
    ping_interval=self.ping_interval,   # 60
    reconnect=self.reconnect_interval   # 0 — disabled
)
```

When the WebSocket drops, there is zero automatic recovery. The entire  
`colab exec` hangs until `--timeout` expires.

### 15.3 WebSocket Ping Gap

**File:** `wsclient.py:496`

`ping_interval=60` sends WebSocket control frames, not TCP payload.  
Chinese carrier NAT devices (5-15 min timeout) may not count WebSocket  
ping frames as activity → connection silently dropped at carrier timeout.

### 15.4 Default Exec Timeout Too Short

**File:** `execution.py:114`

```python
timeout: Annotated[Optional[float], ...] = 10.0  # seconds
```

Any kernel execution lasting >10 seconds without output triggers `TimeoutError`.  
Must always use explicit `--timeout` for training launches.

### 15.5 Kernel Serial Execution

The Jupyter kernel executes code cells **sequentially**. Multiple `colab exec`  
processes can connect simultaneously, but only one executes at a time.  
Additional execs queue. This constrains the relay handoff pattern.

### 15.6 No Proxy Config Passed to WebSocket

**File:** `wsclient.py:1279`, `_app.py:256-274`

`run_forever()` accepts `proxy_type`, `http_proxy_host`, `http_proxy_port`,  
`http_no_proxy` parameters, but `_run_websocket()` passes **none of them**.  
The only proxy resolution is via `websocket._url.get_proxy_info()` reading  
environment variables.

### 15.7 Inline Python via stdin Is Unreliable

Piping Python code via stdin (`echo '...' | colab exec`) can fail for  
multi-line scripts with special characters. Always use `-f <file>` for  
reliable execution.

---

## 16. Dependency Map

```
google-colab-cli
├── typer + click              ← CLI framework
├── pydantic                   ← Request/response model validation
├── requests                   ← REST API (colab.research.google.com)
├── google-auth + google-auth-oauthlib  ← OAuth2 / ADC credential flow
├── google-auth-httplib2       ← (optional ADC dependency)
├── jupyter-kernel-client      ← WebSocket kernel communication
│   ├── websocket-client       ← WebSocket transport
│   ├── jupyter-client         ← Jupyter protocol (session, channels)
│   └── traitlets              ← Configuration system
├── nbformat                   ← Notebook parsing + output saving
├── prompt-toolkit             ← REPL
├── pygments                   ← REPL syntax highlighting
├── rich                       ← REPL styled output
├── filelock                   ← Cross-platform file locking
└── importlib.resources        ← Bundled oauth_config.json, README, SKILL
```

---

## 17. File Size Summary

| File | Lines | Purpose |
|------|-------|---------|
| `commands/utility.py` | 437 | pay, url, log, version, update, whoami |
| `commands/run.py` | 478 | One-shot execution (new+exec+stop) |
| `commands/session.py` | 512 | Session lifecycle + keep-alive daemon |
| `commands/execution.py` | 360 | exec, repl, console |
| `commands/automation.py` | 266 | auth, drivemount, install |
| `commands/files.py` | 205 | ls, rm, upload, download, edit |
| `client.py` | 325 | REST API client |
| `runtime.py` | 263 | WebSocket runtime |
| `converter.py` | 185 | History export |
| `repl.py` | 174 | Interactive REPL |
| `console.py` | 173 | Raw TTY console |
| `auth.py` | 211 | OAuth2/ADC |
| `state.py` | 157 | Persistence |
| `common.py` | 186 | Global state |
| `history.py` | 66 | Event logger |
| `contents.py` | 94 | File I/O |
| `utils.py` | 86 | Helpers |
| `cli.py` | 156 | Entry point |
