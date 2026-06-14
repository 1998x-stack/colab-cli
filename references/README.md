# Colab CLI Reference

Field-tested operational knowledge organized by topic. Each file covers one domain.

## Session Survival

- **[colab-session-survival.md](colab-session-survival.md)** — GPU session death root cause, WebSocket liveness mechanism, relay handoff protocol, combined eval+watchdog pattern, grace period timing, redundant launch strategy

## Network & Proxy

- **[colab-proxy-network.md](colab-proxy-network.md)** — Proxy configs (Config A vs B), REST vs WebSocket paths, China GFW/NAT constraints, WebSocket connection reliability, upload/download behavior

## Operations

- **[colab-operations.md](colab-operations.md)** — File I/O (upload, download, tar, base64 embed), exec patterns, detached bootstrap, multi-account management, cron watchtower

## Training

- **[training-fundamentals.md](training-fundamentals.md)** — LR schedules, weight initialization, NaN diagnosis, gradient clipping, warmup, AMP, overfit testing, CUDA timing
- **[model-specific.md](model-specific.md)** — nanoGPT, nanochat, Transformer IWSLT, text2sql — project-specific gotchas from Colab deployments
- **[cuda-t4-gotchas.md](cuda-t4-gotchas.md)** — T4 limitations, CUDA dark corners, FP16 vs BF16, tensor core utilization, version-specific traps

## Related

- `.claude/skills/colab-cli/references/gotchas.md` — Full operational gotcha catalog (743 lines, canonical source)
- `.claude/skills/colab-cli/references/workflows.md` — Command-level workflows
- `docs/reference/` — Deep-dive analysis docs (source analysis, relay tests, stability analysis)
