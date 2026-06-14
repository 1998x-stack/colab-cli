# google-colab-cli: Core Flow Analysis

**Version:** 0.5.11 | **Date:** 2026-06-14

---

## Flow 1: Session Creation (`colab new --gpu T4 -s mysession`)

### Sequence

```
User                    CLI (session.py)          Colab Backend              Local FS
 │                           │                        │                         │
 │  new --gpu T4 -s mysession│                        │                         │
 ├──────────────────────────►│                        │                         │
 │                           │                        │                         │
 │                           │  assign(uuid4(), GPU, T4)                        │
 │                           ├───────────────────────►│                         │
 │                           │                        │                         │
 │                           │     GET /tun/m/assign?nbh=...&variant=GPU&acc=T4
 │                           │     ← GetAssignmentResponse {token: xsrf}        │
 │                           │                        │                         │
 │                           │     POST /tun/m/assign (X-Goog-Colab-Token: xsrf)
 │                           │     ← PostAssignmentResponse {                   │
 │                           │          endpoint: "gpu-t4-s-kkb-...",           │
 │                           │          runtime_proxy_info: {token, url}        │
 │                           │        }                    │                    │
 │                           │                        │                         │
 │                           │  Pre-flight: keep_alive_assignment(endpoint)     │
 │                           │  → POST colab.pa.googleapis.com/.../KeepAlive    │
 │                           │  ← 403 USER_PROJECT_DENIED                      │
 │                           │  _is_scope_error() → False → passes through      │
 │                           │                        │                         │
 │                           │  SessionState(name, token, url, endpoint, ...)   │
 │                           ├──────────────────────────────────────────────►   │
 │                           │                        │       store.add(s)      │
 │                           │                        │                         │
 │                           │  spawn_keep_alive(endpoint, name)                │
 │                           │  → subprocess.Popen(                             │
 │                           │      "colab keep-alive <endpoint> <name>",       │
 │                           │      start_new_session=True)                     │
 │                           │                        │                         │
 │                           │  store.add(s)  # again with keep_alive_pid       │
 │                           │                        │                         │
 │  "Session READY"          │                        │                         │
 │◄──────────────────────────┤                        │                         │
```

### Critical decision: pre-flight

```python
# session.py:204-220
try:
    state.client.keep_alive_assignment(endpoint)
except ColabRequestError as e:
    if get_status_code(e) == 403 and _is_scope_error(e):
        # Catches: SCOPE_NOT_PERMITTED
        # Misses:  USER_PROJECT_DENIED  ← the actual error
        unassign(endpoint)
        raise typer.Exit(1)
    # "Other failures: don't block session creation"
    # ↑ This comment is wrong — USER_PROJECT_DENIED is not "other",
    #   it's the ONLY error that ever occurs.
```

### Daemon spawn

```python
# session.py:398-421
cmd = [sys.executable, "-m", "colab_cli.cli", "keep-alive", endpoint, session_name]
# Propagates --auth and --config flags so daemon uses same credentials + state file
p = subprocess.Popen(cmd, stdout=DEVNULL, stderr=DEVNULL, stdin=DEVNULL,
                     start_new_session=True)  # detach from parent
```

---

## Flow 2: Keep-Alive Daemon Lifecycle

### Sequence

```
Daemon Process (session.py:keep_alive)      Colab Backend              Local FS
 │                                                │                       │
 │  state.store.get(session_name)                 │                       │
 ├──────────────────────────────────────────────────────────────────────►│
 │  ← SessionState (endpoint, token, ...)                                 │
 │                                                │                       │
 │  while time < 24h:                             │                       │
 │    state.client.keep_alive_assignment(endpoint) │                       │
 ├───────────────────────────────────────────────►│                       │
 │    ← 403 USER_PROJECT_DENIED                   │                       │
 │                                                │                       │
 │    consecutive_4xx = 1                         │                       │
 │    history.log(keep_alive_error)               │                       │
 ├──────────────────────────────────────────────────────────────────────►│
 │                                                │                       │
 │    time.sleep(60)                              │                       │
 │                                                │                       │
 │    state.client.keep_alive_assignment(endpoint) │                       │
 ├───────────────────────────────────────────────►│                       │
 │    ← 403 USER_PROJECT_DENIED                   │                       │
 │                                                │                       │
 │    consecutive_4xx = 2                         │                       │
 │    consecutive_4xx >= 2 → break                │                       │
 │                                                │                       │
 │  history.log(keep_alive_stopped,                │                       │
 │    reason="consecutive_4xx_errors",             │                       │
 │    iterations=2, duration≈61s)                  │                       │
 ├──────────────────────────────────────────────────────────────────────►│
 │                                                │                       │
 │  exit(0)   ← daemon dead after ~61 seconds     │                       │
```

### Outcome

**Every session:** daemon starts → 2 consecutive 403 errors → dies at T+61s.  
**Result:** zero KeepAliveAssignment pings reach Colab backend.  
**Consequence:** session relies entirely on WebSocket liveness (see Flow 3).

---

## Flow 3: Code Execution (`colab exec -s mysession -f train.py --timeout 120`)

### Sequence

```
User          CLI (exec.py)     Runtime (runtime.py)    KernelClient    VM Kernel
 │                │                    │                    │               │
 │  exec -f train.py                 │                    │               │
 ├───────────────►│                    │                    │               │
 │                │                    │                    │               │
 │                │  read train.py     │                    │               │
 │                │  resolve_session() │                    │               │
 │                │  store.get(name)   │                    │               │
 │                │                    │                    │               │
 │                │  ColabRuntime(     │                    │               │
 │                │    url, token,     │                    │               │
 │                │    kernel_id,      │                    │               │
 │                │    session_id)     │                    │               │
 ├───────────────►│                    │                    │               │
 │                │                    │                    │               │
 │                │  execute_code(     │                    │               │
 │                │    "os.chdir(      │                    │               │
 │                │     /content)")    │                    │               │
 ├───────────────►│                    │                    │               │
 │                │                    │                    │               │
 │                │  runtime.kernel_client  ← lazy property               │
 │                ├───────────────────►│                    │               │
 │                │                    │                    │               │
 │                │                    │  KernelClient()    │               │
 │                │                    ├───────────────────►│               │
 │                │                    │                    │               │
 │                │                    │       start()      │               │
 │                │                    ├───────────────────►│               │
 │                │                    │                    │               │
 │                │                    │  POST /api/kernels │  (if no kid)  │
 │                │                    │  ← kernel_id       │               │
 │                │                    │                    │               │
 │                │                    │  WebSocketApp(     │               │
 │                │                    │   wss://.../       │               │
 │                │                    │   api/kernels/kid/ │               │
 │                │                    │   channels?sid=... │               │
 │                │                    │   &token=...)      │               │
 │                │                    ├───────────────────►│               │
 │                │                    │                    │               │
 │                │                    │  run_forever(      │               │
 │                │                    │   ping_interval=60,│               │
 │                │                    │   reconnect=0)     │               │
 │                │                    │                    │               │
 │                │                    │  Channels: shell, iopub, stdin, hb, control
 │                │                    ├───────────────────►│               │
 │                │                    │                    │               │
 │                │  execute_code(     │                    │               │
 │                │    train.py body,  │                    │               │
 │                │    output_hook=    │                    │               │
 │                │    display_output, │                    │               │
 │                │    timeout=120)    │                    │               │
 ├───────────────►│                    │                    │               │
 │                │                    │                    │               │
 │                │                    │  execute_interactive(code)         │
 │                │                    ├───────────────────►│               │
 │                │                    │                    │               │
 │                │                    │    execute_request  │               │
 │                │                    │    (shell channel)  │               │
 │                │                    ├───────────────────►├──────────────►│
 │                │                    │                    │               │
 │                │                    │    IOPub messages   │               │
 │                │                    │◄───────────────────┤◄──────────────┤
 │                │                    │    (stream,         │               │
 │                │                    │     display_data,   │               │
 │                │                    │     execute_result, │               │
 │                │                    │     error)          │               │
 │                │                    │                    │               │
 │                │  display_output(o) │                    │               │
 │                │◄───────────────────┤                    │               │
 │                │                    │                    │               │
 │  stdout/stderr  │                    │                    │               │
 │◄───────────────┤                    │                    │               │
 │                │                    │                    │               │
 │                │  runtime.stop()    │                    │               │
 │                ├───────────────────►│                    │               │
 │                │                    │  stop_channels()   │               │
 │                │                    │  kernel_socket.close()
 │                │                    │  (WebSocket closed) │               │
 │                │                    │                    │               │
 │                │  store.add(s)      │                    │               │
 │                │  (s.running=None)  │                    │               │
```

### Key details

**Kernel connection retry (runtime.py:92-155):**
```python
for i in range(3):
    try:
        self._kernel_client = jupyter_kernel_client.KernelClient(...)
        self._kernel_client.start()     # Connect WebSocket
        self._apply_ws_hook()           # Install Drive auth interceptor
        break
    except (ReadTimeout, ConnectTimeout) as e:
        if i < 2:
            time.sleep(backoff ** (i + 1))   # 2s, 4s
        else:
            raise
```

**WebSocket hook (runtime.py:46-87):**
Intercepts `colab_request` messages on the kernel IOPub channel.  
Used by `drivemount` to complete OAuth without browser interaction.  
Returns `True` to prevent the message from reaching the default handler.

**Output hook (runtime.py:204-221):**
```python
if output_hook:
    # Streaming: execute_interactive → hook called per IOPub message
    reply = self.kernel_client.execute_interactive(code, output_hook=wrapped_hook)
else:
    # Batch: execute → returns all outputs at once
    reply = self.kernel_client.execute(code)
```

**exec default timeout:** 10s (`execution.py:114`). This is a wall-clock budget  
that shrinks on every poll iteration. A single >10s quiet period in the output  
stream triggers `TimeoutError` even if the kernel is still executing.

---

## Flow 4: File Upload (`colab upload local.py /content/train.py`)

### Sequence

```
CLI (files.py)          ContentsClient              VM (Jupyter Contents API)
 │                           │                              │
 │  upload local.py          │                              │
 │  /content/train.py        │                              │
 ├──────────────────────────►│                              │
 │                           │                              │
 │                           │  read local.py → bytes       │
 │                           │  base64.encode(content)      │
 │                           │                              │
 │                           │  PUT /api/contents/content/train.py
 │                           │  ?authuser=0                 │
 │                           │  &colab-runtime-proxy-token=<jwt>
 │                           │                              │
 │                           │  Body: {                     │
 │                           │    "name": "train.py",       │
 │                           │    "path": "/content/train.py",│
 │                           │    "type": "file",           │
 │                           │    "format": "base64",       │
 │                           │    "content": "<b64>",       │
 │                           │    "chunk": 1                │
 │                           │  }                           │
 ├──────────────────────────►├─────────────────────────────►│
 │                           │                              │
 │                           │  ← 201 Created               │
 │                           │◄─────────────────────────────┤
 │  "Uploaded"               │                              │
 │◄──────────────────────────┤                              │
```

**Authentication:** The JWT `runtime_proxy_token` from session creation is passed  
as a query parameter, not a header. This token is an ES256 JWT with audience =  
the endpoint ID, valid for 3600 seconds.

**Note:** `upload` goes through the WebSocket path (`*.prod.colab.dev`), NOT  
through `colab.research.google.com`. This means uploads can fail when the  
WebSocket path is unstable.

---

## Flow 5: Directory Download Workaround

```
CLI (files.py)         ContentsClient              VM
 │                           │                      │
 │  download /content/out/   │                      │
 │  → IsADirectoryError      │                      │
 │                           │                      │
 │  Workaround:              │                      │
 │  1. exec: tar -czf        │                      │
 │     /tmp/out.tar.gz       │                      │
 │     -C /content out/      │                      │
 │                           │                      │
 │  2. download /tmp/out.tar.gz → local out.tar.gz  │
 │                           │                      │
 │  3. local: tar -xzf out.tar.gz                   │
```

`ContentsClient.download()` checks `data.get("type")` — if `"directory"`,  
raises `IsADirectoryError`. The Jupyter Contents API only serves single files.

---

## Flow 6: Session Sync & Pruning

### When it runs

Called at the start of every `colab exec`, `colab status`, `colab sessions`,  
and `colab stop` (via `state.resolve_session()` or `state.sync_sessions()`).

### Sequence

```
State.sync_sessions()                     Colab Backend
 │                                              │
 │  local_sessions = store.list()               │
 │  (from ~/.config/colab-cli/sessions.json)    │
 │                                              │
 │  assignments = client.list_assignments()     │
 ├─────────────────────────────────────────────►│
 │  ← [ListedAssignment(endpoint=...), ...]     │
 │                                              │
 │  active_endpoints = {a.endpoint for a in assignments}
 │                                              │
 │  for name, s in local_sessions:              │
 │    if s.endpoint NOT in active_endpoints:    │
 │      prune_session(name)                     │
 │        → kill_process(s.keep_alive_pid)       │
 │        → store.remove(name)                  │
 │        → history.log(session_terminated, "pruned")
 │                                              │
 │  return local_sessions, assignments          │
```

**Caching:** `self._sessions` caches the result. Set to `None` to force re-sync.

---

## Flow 7: `colab run` — One-Shot Execution

```
User            CLI (run.py)          Session       Runtime        VM
 │                  │                    │             │             │
 │  run --gpu T4    │                    │             │             │
 │  train.py --lr 0.1                   │             │             │
 ├─────────────────►│                    │             │             │
 │                  │                    │             │             │
 │                  │  validate: train.py exists locally               │
 │                  │                    │             │             │
 │                  │  assign(GPU, T4)   │             │             │
 │                  │  → session created │             │             │
 │                  │                    │             │             │
 │                  │  pre-flight keep-alive           │             │
 │                  │  spawn_keep_alive()              │             │
 │                  │                    │             │             │
 │                  │  ColabRuntime()    │             │             │
 │                  │  execute_code(     │             │             │
 │                  │    os.chdir(       │             │             │
 │                  │    /content))      │             │             │
 │                  ├───────────────────►│             │             │
 │                  │                    │             │             │
 │                  │  payload = _build_script_payload(train.py, ["--lr","0.1"])
 │                  │  = "import sys, warnings\n" +                    │
 │                  │    "sys.argv = ['train.py', '--lr', '0.1']\n" + │
 │                  │    "__name__ = '__main__'\n" +                   │
 │                  │    "warnings.filterwarnings(...)\n" +            │
 │                  │    train.py_body                                 │
 │                  │                    │             │             │
 │                  │  execute_code(payload, timeout=30)               │
 │                  ├───────────────────►│             │             │
 │                  │                    │  exec       │             │
 │                  │                    ├────────────►│             │
 │                  │                    │  outputs    │             │
 │                  │                    │◄────────────┤             │
 │                  │                    │             │             │
 │                  │  _exit_code_from_outputs(outputs)                │
 │                  │  sys.exit(0) → 0, other error → 1               │
 │                  │                    │             │             │
 │                  │  runtime.stop()    │             │             │
 │                  │                    │             │             │
 │                  │  if not keep:      │             │             │
 │                  │    _teardown():    │             │             │
 │                  │     kill_process(keep_alive_pid)                │
 │                  │     runtime.stop(shutdown_kernel=True)          │
 │                  │     client.unassign(endpoint)                   │
 │                  │     store.remove(name)                          │
 │                  │                    │             │             │
 │  exit(code)       │                    │             │             │
 │◄─────────────────┤                    │             │             │
```

**Exit code mapping:**
```python
# run.py:142-160
def _exit_code_from_outputs(outputs):
    code = 0
    for o in outputs:
        if o.get("output_type") != "error": continue
        if _is_systemexit(o):
            ec = _systemexit_code(o)   # None/0→0, int→int, str→1
            code = ec if ec != 0 else code
        else:
            return 1                    # Uncaught exception → exit 1
    return code
```

---

## Flow 8: WebSocket Drop & Recovery (The Relay Pattern)

### What happens when WebSocket drops

```
KernelWebSocketClient._run_websocket()
  │
  │  run_forever(ping_interval=60, reconnect=0)
  │    │
  │    │  ... 8-15 minutes pass ...  ← NAT/GFW timeout
  │    │
  │    ▼  WebSocket connection lost (no FIN/RST — silent drop)
  │
  │  run_forever() returns  (or raises, depending on drop type)
  │
  │  connection_thread exits
  │
  colab exec hangs
    │
    │  execute_interactive() waiting for reply...
    │  poll loop: check for new messages, timeout shrinks
    │
    ▼  timeout expires → TimeoutError
```

**No automatic reconnect** because `reconnect_interval=0`.

### Recovery: Relay Handoff Pattern

```
Timeline:
  T+0      colab exec ws-1 → connects, executes watchdog.py
  T+6      colab exec ws-2 → connects, code QUEUED (kernel serial execution)
  T+7      ws-1 exits → ws-2 starts executing from queue (~5s gap)
  T+13     colab exec ws-3 → connects, code QUEUED
  T+14     ws-2 exits → ws-3 starts executing
```

**Why this works:**
1. Each WebSocket connection is independent (different `colab exec` process)
2. Kernel serial execution means next watchdog starts within ~5 seconds
3. 5-second gap is well within the ~2-3 minute session grace period
4. Even if one WebSocket drops early, the next one takes over

**Constraint:** Each watchdog must run <8 minutes (inside China WebSocket  
stability window) but the relay chain can sustain sessions indefinitely.

---

## Flow 9: Session Termination (`colab stop -s mysession`)

```
CLI (session.py:stop)           Daemon        Runtime         Backend     FS
 │                                 │              │               │         │
 │  resolve_session("mysession")   │              │               │         │
 │  store.get("mysession")         │              │               │         │
 │                                 │              │               │         │
 │  kill_process(keep_alive_pid)   │              │               │         │
 ├────────────────────────────────►│              │               │         │
 │  SIGTERM → daemon exits         │              │               │         │
 │                                 │              │               │         │
 │  ColabRuntime(url, token, kid)  │              │               │         │
 │  runtime.stop(shutdown_kernel=True)            │               │         │
 ├───────────────────────────────────────────────►│               │         │
 │                                 │  stop_channels()              │         │
 │                                 │  kernel_socket.close()        │         │
 │                                 │  shutdown_kernel(now=True)    │         │
 │                                 │  → DELETE /api/kernels/kid    │         │
 │                                 │              │               │         │
 │  client.unassign(endpoint)      │              │               │         │
 ├───────────────────────────────────────────────────────────────►│         │
 │                                 │              │  GET /tun/m/unassign/ep  │
 │                                 │              │  ← {token: xsrf}│         │
 │                                 │              │  POST (with xsrf header)  │
 │                                 │              │  ← OK          │         │
 │                                 │              │               │         │
 │  store.remove(name)             │              │               │         │
 ├──────────────────────────────────────────────────────────────────────────►│
 │                                 │              │               │   delete │
 │  history.log(session_terminated, "user_requested")              │         │
 ├──────────────────────────────────────────────────────────────────────────►│
 │                                 │              │               │  append │
 │  "Session terminated."          │              │               │         │
```

---

## 10. State Transitions

```
SessionState.running field:

  None ──► "exec(train.py)" ──► None
  None ──► "repl"            ──► None
  None ──► "console"         ──► None
  None ──► "automation(drivemount)" ──► None

SessionState.last_execution:
  (filename, cell_id, "2026-06-14 08:10:41")

Session lifecycle:
  [Created] ──► [Active] ──► [Pruned]   (server reclaimed)
  [Created] ──► [Active] ──► [Stopped]  (user requested)
```

---

## Summary: Critical Paths

| Flow | Critical file | Key function | Failure mode |
|------|-------------|-------------|--------------|
| Session create | `session.py:112-246` | `new()` | 403 keep-alive (silent) |
| Keep-alive | `session.py:424-502` | `keep_alive()` | 2× 403 → exit at 61s |
| Code exec | `runtime.py:164-246` | `execute_code()` | WS drop → TimeoutError |
| File upload | `contents.py:56-72` | `upload()` | WS path unstable |
| File download | `contents.py:74-90` | `download()` | No directory support |
| Session sync | `common.py:81-112` | `sync_sessions()` | Prunes stale sessions |
| Relay handoff | `execution.py` + `runtime.py` | `exec_command()` + `kernel_client` | Kernel serial queue |
| Session stop | `session.py:351-380` | `stop()` | Best-effort cleanup |
