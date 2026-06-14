# Colab Proxy & Network

Proxy configuration and network behavior from mainland China.

## Two Independent Network Paths

Colab CLI uses two separate transport layers with different proxy behavior:

| Path | Protocol | Library | Domain | Proxy support |
|------|----------|---------|--------|--------------|
| REST (new, stop, keep-alive) | HTTPS | `requests` | `colab.pa.googleapis.com` | Auto-detects `HTTP_PROXY`/`HTTPS_PROXY` |
| REST (upload, download) | HTTPS | `requests` | `*.prod.colab.dev` | Same as above, affected by `no_proxy` |
| WebSocket (exec, repl) | WSS | `websocket-client` | `*.prod.colab.dev` | Does NOT pass proxy params; reads `https_proxy` env |

Key distinction: upload/download use REST (not WebSocket) despite sharing `*.prod.colab.dev` domain. They survive WebSocket drops.

## Proxy Configs

### Config B — HTTP CONNECT Tunnel (Recommended)

Both paths through proxy. More reliable for full-session workflows (upload→exec→download).

```bash
export HTTPS_PROXY=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export ALL_PROXY=socks5://127.0.0.1:7890
```

### Config A — SOCKS5 REST + WebSocket Direct (Fallback)

REST through SOCKS5, WebSocket bypasses proxy. Try if Config B fails for WebSocket connections.

```bash
export HTTPS_PROXY=socks5://127.0.0.1:7890
export HTTP_PROXY=socks5://127.0.0.1:7890
export no_proxy="*.colab.dev,*.prod.colab.dev,localhost,127.0.0.1"
```

Config A's `no_proxy` exclusion for `*.prod.colab.dev` can cause SSL/EOF errors on upload/download REST calls. Start with Config B, flip if WebSocket connections fail.

## China WebSocket Constraints

Three layers cause WebSocket drops at 5-15 min:

| Layer | Mechanism | Timeout |
|-------|-----------|---------|
| Carrier NAT | NAT table entries expire; WebSocket ping frames (2-byte TCP segments) may not reset idle timer | 5-15 min |
| GFW | Stateful inspection terminates encrypted tunnels to Google IPs | 10-15 min |
| Application | `ping_interval=60` too sparse; `reconnect_interval=0` (no auto-recovery) | — |

**Key insight:** WebSocket ping frames (opcode 0x9) are 2-byte TCP segments with no application payload. Many NAT implementations don't count them as activity. Application-level heartbeats (nvidia-smi checks, print statements) generate real TCP payload and reliably reset NAT timers.

## Connection Reliability

From live tests (2026-06-14): **~60% success rate per WebSocket connection attempt.** Failed attempts die at the handshake/chdir stage with `TimeoutError: Timeout waiting for reply` — unrelated to session health.

## SSL Errors

`SSLError: UNEXPECTED_EOF_WHILE_READING` happens occasionally. Usually transient — retry 2-3 times. Session is often still alive.

## Proxy Not Running

`ProxyError: Unable to connect to proxy` means Clash/Meta at `127.0.0.1:7890` is down. Check `ps aux | grep clash`. Restart if needed.

## GPU Provisioning Errors

| Config | Error | Meaning |
|--------|-------|---------|
| A (SOCKS5 + no_proxy) | 503 Service Unavailable | Ambiguous — proxy issue OR GPU exhaustion |
| B (HTTP CONNECT) | 412 Precondition Failed / TooManyAssignmentsError | Genuine GPU quota exhaustion |

**Procedure:** If Config A returns 503, flip to Config B. If Config B returns 412, GPU exhausted — switch accounts or wait 12-24h.
