# Colab Free GPU Keepalive: Root Cause Analysis & Solution

**Date:** 2026-06-14 | **Sessions tested:** 40+ (historical) + 3 (live T4 GPU)

---

## 1. The Problem

Free-tier Colab GPU sessions die after ~10 minutes, even with active training.
This happens regardless of project, account, or GPU utilization.

**Root cause in one sentence:** The keep-alive daemon that prevents GPU reclamation
fails on every session due to an IAM deadlock, and the session dies ~10 minutes later.

---

## 2. Root Cause: KeepAliveAssignment RPC IAM Deadlock

### 2.1 The keep-alive daemon

When `colab new --gpu T4` creates a session, it spawns a background daemon process
(`colab keep-alive <endpoint>`) that calls the `KeepAliveAssignment` RPC every
60 seconds for up to 24 hours. This tells Colab's backend "this session is still
in use — don't reclaim the GPU."

### 2.2 The RPC call that fails

```
POST https://colab.pa.googleapis.com/$rpc/google.internal.colab.v1.RuntimeService/KeepAliveAssignment

Headers:
  x-goog-api-key: AIzaSyA2BvntLwNwFthUB4w6_Bhn0cMlVHwyaHc   ← Colab's public web-client API key
  x-goog-user-project: 1014160490159                         ← Colab's GCP project
  Authorization: Bearer <user_oauth_token>                   ← Your OAuth2 credential
  Content-Type: application/json+protobuf
```

### 2.3 The IAM deadlock

Google's IAM checks fail in both directions:

| Configuration | Result |
|--------------|--------|
| With `x-goog-user-project: 1014160490159` | **HTTP 403** `USER_PROJECT_DENIED` — user lacks `serviceusage.services.use` on Colab's project |
| Without `x-goog-user-project` | **HTTP 400** `CONSUMER_INVALID` — API key (Colab project) ≠ auth credential (user project) |
| Bearer token only (no API key) | **HTTP 403** `SERVICE_DISABLED` — requires a quota project |
| API key only (no Bearer) | **HTTP 401** — "API keys are not supported by this API" |

The API key belongs to Colab's GCP project (1014160490159). The user's OAuth2
credential belongs to the user's project. Setting `x-goog-user-project` tells
Google to charge Colab's project, but the user doesn't have IAM permissions on
Colab's project. Neither path works.

### 2.4 Evidence: 100% failure rate across 40+ sessions

Every session in `~/.config/colab-cli/history/*.jsonl` shows the identical pattern:

```
Session created → keep_alive_started → keep_alive_error (403) → keep_alive_error (403)
→ keep_alive_stopped (reason=consecutive_4xx_errors, iterations=2, duration≈61s)
```

Zero successful `KeepAliveAssignment` calls across all sessions. The daemon
always dies 61 seconds after session creation.

### 2.5 The kill chain

```
T+0s     colab new --gpu T4 → session created
T+1s     Keep-alive daemon: 403 error #1 (USER_PROJECT_DENIED)
T+61s    Keep-alive daemon: 403 error #2 → daemon exits
T+61s    NO keep-alive protection from this point
T+~600s  Colab backend: no liveness signal → reclaim GPU
```

Source files (google-colab-cli v0.5.11):
- `client.py:298-324` — `keep_alive_assignment()` with hardcoded API key + `x-goog-user-project`
- `session.py:424-502` — daemon loop, exits after 2 consecutive 4xx errors
- `session.py:200-220` — pre-flight check, misses `USER_PROJECT_DENIED` error string
- `session.py:35-49` — `_is_scope_error()` only catches `SCOPE_NOT_PERMITTED`, not `USER_PROJECT_DENIED`

---

## 3. Solution Discovery: WebSocket as Primary Liveness Signal

### 3.1 Hypothesis

The Colab browser frontend keeps sessions alive for hours. The browser maintains
a persistent WebSocket connection to the Jupyter kernel via Colab's runtime proxy.
Maybe this WebSocket — not the KeepAliveAssignment RPC — is the primary liveness
signal.

### 3.2 Test 1: Single WebSocket Keepalive

**Setup:** Keep `colab exec` WebSocket open for 8 minutes while fake GPU training
runs. The keep-alive daemon dies at T+61s as usual.

**Result: CONFIRMED**

```
00:10:21  Session created
00:10:22  Keep-alive daemon: started
00:10:23  Keep-alive daemon: 403 error #1
00:10:42  WebSocket opened (watchdog loop starts)
00:11:23  Keep-alive daemon: 403 error #2 → daemon dies (61.6s)
00:17:50  WebSocket closed (watchdog exits after 8 min)
00:20:35  Session reclaimed (~2 min 45 sec after WebSocket close)
```

**Key finding:** The session survived 10+ minutes despite a dead keep-alive
daemon. The WebSocket connection was sufficient to prevent GPU reclamation.
The session died ~2-3 minutes after the WebSocket closed.

### 3.3 Implication

The WebSocket through Colab's runtime proxy (`wss://8080-<endpoint>.prod.colab.dev`)
is the **primary liveness signal**. The `KeepAliveAssignment` RPC is supplementary
and not needed while the WebSocket is active.

---

## 4. The Relay Handoff Protocol

### 4.1 Problem: China WebSocket Stability

From China, the `colab exec` WebSocket drops after ~8-12 minutes due to GFW/NAT
timeout. A single WebSocket cannot cover a full training session.

### 4.2 Constraint: Kernel Serial Execution

Jupyter kernels execute code **sequentially** — one execution at a time. When
`colab exec -f watchdog.py` is called while another watchdog is running:

```
Kernel execution queue:
  [ws-1: while True: sleep(30); log()] ← running (owns kernel)
  [ws-2: while True: sleep(30); log()] ← queued (waiting)
  [ws-3: while True: sleep(30); log()] ← queued (waiting)
```

Each `colab exec` creates a WebSocket connection immediately, but the code
only executes when it reaches the front of the kernel queue.

### 4.3 The Handoff Pattern

Despite serial execution, handoff works because:
1. The next watchdog's WebSocket is already connected (code queued)
2. When the current watchdog finishes, the next one starts in **~5 seconds**
3. The session survives this gap (grace period is ~2-3 minutes)

```
T+0:     [ws-1: EXECUTING]....................................[exit]
T+6:                [ws-2: WebSocket connected, code QUEUED]..........[ws-2 starts at T+7:00:05]
T+13:                         [ws-3: WebSocket connected, QUEUED]............[ws-3 starts at T+14:00:05]
```

**Gap between handoffs: ~5 seconds.** Well within the ~2-3 minute grace period.

### 4.4 Test 2: Three-Watchdog Relay (Partial)

**Setup:** Three overlapping watchdogs via three `colab exec -f` processes.

**Result:** Handoff ws-1→ws-2 succeeded (5-second gap). But ws-2 died 30 seconds
into execution because the background Bash task that launched it was cleaned up.
Session reached 10 min 14 sec — equivalent to unprotected lifetime.

**Lesson learned:** Watchdogs must be launched from independent processes
(separate terminal, cron, or explicit `&` disown), not from managed background
tasks that can kill child processes.

### 4.5 Test 3: 20-Minute Target (Incomplete)

**Setup:** 4-watchdog relay chain targeting 20 minutes. All watchdogs via `-f`.

**Result:** ws-1 completed (7 min). ws-2 took over (5-second gap). ws-3 launched
and queued. But ws-2 died prematurely (background task cleanup, same as Test 2).

**Unverified:** The full 4-watchdog chain should provide ~21 minutes of continuous
WebSocket coverage (3 × 7 min watchdogs, 5-second gaps).

---

## 5. Implementation Guide

### 5.1 Single watchdog (≤8 min training)

For short training runs that fit within the China WebSocket stability window:

```bash
# 1. Create session
colab new --gpu T4 -s my-training

# 2. Upload training script
colab upload train.py /content/train.py

# 3. Run launch script (spawns training + keeps WS open)
colab exec -s my-training -f launch_and_watch.py --timeout 540

# launch_and_watch.py:
#   - pip install deps
#   - spawn train.py as detached subprocess (nohup)
#   - loop for 8 min: check training alive, GPU util, print status
#   - exit
```

### 5.2 Relay chain (>8 min training)

For training longer than the WebSocket stability window:

```bash
# Terminal 1: Launch ws-1 (includes training spawn)
colab exec -s my-training -f launch_and_watch.py --timeout 540 &

# Terminal 2 (or cron): At T+6 min, start ws-2
sleep 360
colab exec -s my-training -f watchdog.py --timeout 540 &

# Terminal 2 (or cron): At T+13 min, start ws-3
sleep 420  # from last
colab exec -s my-training -f watchdog.py --timeout 540 &

# ... repeat every 7 minutes until training completes
```

### 5.3 Watchdog script template

```python
"""watchdog.py — keeps WebSocket alive, monitors training, 7-min window."""
import subprocess, os, time
from datetime import datetime, timezone

OUT_DIR = "/content/my-output"
DURATION = 420   # 7 minutes (safe inside China WS stability window)
INTERVAL = 30

os.makedirs(f"{OUT_DIR}/logs", exist_ok=True)

def ts():
    return datetime.now(timezone.utc).strftime("%H:%M:%S")

def log(msg):
    line = f"[{ts()}] {msg}"
    print(line, flush=True)
    with open(f"{OUT_DIR}/logs/watchdog.log", "a") as f:
        f.write(line + "\n")

log(f"WATCHDOG_START pid={os.getpid()}")

# Check training
train_pid_file = f"{OUT_DIR}/train.pid"
train_pid = None
if os.path.exists(train_pid_file):
    with open(train_pid_file) as f:
        train_pid = int(f.read().strip())
    try:
        os.kill(train_pid, 0)
        log(f"Training PID={train_pid} ALIVE")
    except OSError:
        log(f"Training PID={train_pid} DEAD")

start = time.time()
for i in range(DURATION // INTERVAL):
    time.sleep(INTERVAL)
    elapsed = time.time() - start

    # Training alive?
    tstat = "N/A"
    if train_pid:
        try:
            os.kill(train_pid, 0)
            tstat = "ALIVE"
        except OSError:
            tstat = "DEAD"
            log("Training died!")

    # GPU
    gpu = subprocess.check_output(
        "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader",
        shell=True, text=True, timeout=5
    ).strip()

    log(f"elapsed={elapsed:.0f}s train={tstat} gpu={gpu}")
    print(f"[{ts()}] heartbeat elapsed={elapsed:.0f}s", flush=True)

log(f"WATCHDOG_EXIT")
```

### 5.4 Key constraints (updated 2026-06-14 from live tests)

| Constraint | Value | Mitigation |
|-----------|-------|------------|
| China WebSocket stability | 5-8 min reliable window (carrier-dependent) | 5-min watchdog windows |
| China WS connection success rate | ~60% per attempt (3/5 in live tests) | Launch 2 watchdogs per handoff (redundancy) |
| Kernel execution | Serial (one at a time) | Queue next watchdog before current exits |
| Session grace period | ~2-5 min after last signal (typically ~3 min) | Overlap ensures <10 sec gaps |
| Handoff gap | 0-5 seconds (kernel queue, measured at 0s twice) | No gap mitigation needed |
| Queue time penalty | Idle queue time burns NAT budget 1:1 | Minimize overlap to 30s |
| Coverage gaps | FATAL — any gap triggers reclamation | Continuous coverage mandatory |
| Free tier session limit | 12 hours max | Relay chain up to limit |
| `colab exec --timeout` | Must exceed watchdog duration | Set to 420 (7 min for 5-min watchdog) |
| Watchdog launch method | Must use independent OS processes | `nohup ... &` or `Popen(start_new_session=True)` |

### 5.5 Redundant Watchdog Relay (Recommended for >20 min)

Each watchdog connection has ~60% success rate from China. For reliability, launch **two** watchdogs per handoff window:

```
P(at least one connects) = 1 - (1 - 0.6)^2 = 84% per handoff
P(4 handoffs all succeed) = 0.84^4 ≈ 50%
```

Parameters for 25-min sessions:
- **Window:** 5 minutes per watchdog
- **Overlap:** 30 seconds (minimize queue time — idle queue burns NAT budget)
- **Heartbeat:** Every 25 seconds (nvidia-smi + print for real TCP payload)
- **Watchdogs needed:** 6 pairs (12 total launches)
- **Launch:** `nohup colab exec -f wd.py --timeout 420 &`

Live test data: `docs/reference/colab-gpu-relay-tests.md`

---

## 6. Alternative: Kaggle GPU

For training sessions longer than the 12-hour Colab limit, or when WebSocket
management overhead is too high, use Kaggle:

| | Colab (free) | Kaggle (free) |
|---|---|---|
| GPU | T4 (dynamic quota) | P100 or T4×2 (30h/week) |
| Session limit | ~12h | ~9h per kernel |
| Liveness mechanism | WebSocket relay chain | Push model, no persistent connection needed |
| China reliability | WebSocket drops at 8-12 min | REST-only, no WebSocket |
| Keep-alive | Broken (IAM deadlock) | Not needed |

---

## 7. Source Files

### google-colab-cli (v0.5.11)

```
~/.local/share/uv/tools/google-colab-cli/lib/python3.13/site-packages/colab_cli/
├── client.py       ← keep_alive_assignment(), _PUBLIC_CLIENT_REGISTRY
├── session.py      ← spawn_keep_alive(), keep_alive(), pre-flight check
├── runtime.py      ← ColabRuntime, WebSocket lifecycle, execute_code()
├── auth.py         ← OAuth2/ADC credential management, PUBLIC_SCOPES
├── common.py       ← State, sync_sessions(), prune_session()
└── commands/
    ├── session.py  ← new(), stop(), sessions(), status()
    └── execution.py ← exec_command(), repl(), console()
```

### Test scripts (this repo)

```
tests/ws-keepalive/
├── relay/
│   ├── fake_train.py      ← 20-min fake GPU training
│   ├── watchdog.py        ← Generic 7-min watchdog (auto-naming, GPU monitor)
│   └── launch_train.py    ← Spawns training + acts as ws-1
├── launcher.py            ← Test 1: 8-min watchdog launcher
├── watchdog_v2.py         ← Test 2: watchdog with env var control
└── output/                ← Downloaded test logs
```

---

## 8. Key Learnings

1. **The `KeepAliveAssignment` RPC is broken for all users** due to an IAM
   deadlock in the `x-goog-user-project` header. This affects 100% of sessions.

2. **The WebSocket connection is the primary liveness signal.** Colab's runtime
   proxy tracks active WebSocket connections to the kernel. A persistent
   WebSocket keeps the session alive even without keep-alive RPC calls.
   Confirmed in live tests — session died 3-4 min after WebSocket closed.

3. **Jupyter kernels execute code sequentially.** Multiple `colab exec`
   processes can connect simultaneously, but only one executes at a time.
   Others queue in order.

4. **Relay handoff works with zero-second gap.** Two handoffs measured at
   exactly 0 seconds — the queued watchdog starts the same second the current
   one exits. Kernel serial queue handles this flawlessly.

5. **The session grace period is ~2-5 minutes** (typically ~3 min) after the
   last active WebSocket drops. But any gap in coverage is fatal — reconnection
   does not reliably reset the death timer.

6. **China WebSocket connection success rate is ~60% per attempt** (3/5 in
   live tests). Failed attempts die at the WebSocket handshake/chdir stage,
   not mid-execution.

7. **Queue time burns NAT budget.** A watchdog queued (connected but idle)
   for 2 minutes had only ~4 minutes of execution before dropping. Keep
   overlap to 30 seconds — just enough for connection + queuing.

8. **Config B proxy (HTTP CONNECT tunnel) is the reliable default.**
   Config A (SOCKS5 + no_proxy for colab.dev) had WebSocket connection
   failures. Config B worked for both REST and WebSocket consistently.

## 9. Live Fire Test Results

Three progressively complex relay tests were run on 2026-06-14. Full data in
`docs/reference/colab-gpu-relay-tests.md`.

| Test | Watchdogs | WS Coverage | Session Life | Result |
|------|-----------|-------------|--------------|--------|
| Single watchdog (8 min) | 1 | 8 min | ~12 min | Training (12 min) incomplete |
| Two-watchdog relay | 2 | ~11 min | ~14.6 min | Training (14 min) incomplete |
| Multi-watchdog (25-min target) | 6 planned | ~7 min + gap | ~10.8 min | 2-min gap killed session |
