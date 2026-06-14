# Colab Session Survival

How free-tier GPU sessions actually die, and how to keep them alive.

## Root Cause

The keep-alive daemon (`KeepAliveAssignment` RPC to `colab.pa.googleapis.com`) is **permanently broken** due to an IAM deadlock. The `x-goog-user-project: 1014160490159` header triggers a 403 `USER_PROJECT_DENIED` for all users. The daemon exits at T+61s every session — 100% failure rate across 40+ observed sessions.

The **WebSocket connection** through Colab's runtime proxy (`wss://*.prod.colab.dev`) is the actual liveness signal. Colab's backend tracks active WebSocket connections and reclaims VMs after the last connection drops.

## Kill Chain

```
T+0s     Session created, keep-alive daemon spawned
T+1s     KeepAliveAssignment → 403 (IAM deadlock)
T+61s    Second 403 → daemon exits (2 consecutive 4xx)
T+~600s  Backend: no liveness for ~9 min → reclaim GPU
```

## Verified Metrics (2026-06-14)

| Metric | Value |
|--------|-------|
| Grace period (last WS close → death) | 2-5 min (typically ~3 min) |
| China WS connection success rate | ~60% per attempt (3/5 in tests) |
| China WS stable execution window | 5-8 min (carrier-dependent) |
| Handoff gap (kernel serial queue) | 0 seconds (measured twice) |
| Queue time penalty | 1:1 (1 min queued = 1 min less execution budget) |
| Coverage gaps | **FATAL** — reconnection does NOT reset death timer |

## Strategy 1: Relay Handoff Chain

For sessions requiring >8 min of WebSocket coverage. Jupyter kernel executes code serially — the next watchdog's code queues behind the current one, starts the same second.

```
T+0:00   [ws-1: EXECUTING]........................[exits T+5:00]
T+4:30        [ws-2: WS connected, code QUEUED]..[starts T+5:00:00]
T+9:00              [ws-3: QUEUED]...............[starts T+10:00:00]
```

**Parameters:**
- Watchdog window: 5 minutes
- Overlap: 30 seconds (minimize queue time — idle queue burns NAT budget)
- Heartbeat: every 25s (nvidia-smi + print = real TCP payload)
- Launch: `nohup colab exec -f wd.py --timeout 420 &`
- Redundancy: 2 watchdogs per handoff (84% success rate)

**Probability:** P(4 handoffs succeed) = 0.84^4 ≈ 50%

## Strategy 2: Combined Eval+Watchdog

For workflows where training spawns detached and eval follows. A single script keeps the WebSocket alive from launch to completion — zero gaps.

**Pattern (eval_and_watch.py):**
```python
# Wait for training with heartbeats
while not training_done:
    print(f"[{ts()}] waiting...", flush=True)
    time.sleep(15)

# Load model with heartbeats
print(f"[{ts()}] loading model...", flush=True)

# Eval loop with heartbeats every 5 examples
for i, ex in enumerate(test_data):
    # ... eval ...
    if (i + 1) % 5 == 0:
        print(f"[{ts()}] eval {i+1}/N | acc={acc:.3f}", flush=True)
```

**Why it works:** Single WebSocket connection, active from start to finish. No gaps, no queue time penalty. Heartbeats every 15-25s generate TCP payload to reset NAT timers.

**Verified:** text2sql_finetune session survived 10+ min, eval completed 100/100 examples (2026-06-14).

**Redundancy:** Launch 2 instances, 30s apart:
```bash
nohup colab exec -s "$S" -f eval_and_watch.py --timeout 600 > /tmp/eval1.log 2>&1 &
sleep 30
nohup colab exec -s "$S" -f eval_and_watch.py --timeout 600 > /tmp/eval2.log 2>&1 &
```

## Strategy 3: SSH Tunnel (Highest Reliability)

Eliminates GFW and carrier NAT by tunneling through a VPS:

```bash
ssh -D 7892 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 user@vps
export HTTPS_PROXY=socks5://127.0.0.1:7892
```

Under SSH tunnel, single WebSocket survives hours. GFW sees SSH traffic to VPS, not WSS to Google IPs. Cost: ~$5/month VPS.

## Deployment Checklist

- [ ] Config B proxy (`HTTPS_PROXY=http://`, `ALL_PROXY=socks5://`)
- [ ] Training spawned detached (`start_new_session=True`)
- [ ] WebSocket launched immediately after (no gap)
- [ ] Redundant launch for reliability (2 instances)
- [ ] Heartbeats every 15-25s (real TCP payload, not just WebSocket ping)
- [ ] `colab exec --timeout` exceeds expected duration by 2 min
- [ ] Cron watchtower for REST-based fallback monitoring
