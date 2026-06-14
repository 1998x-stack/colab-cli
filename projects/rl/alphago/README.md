# AlphaGo Zero — 9×9 Go

Self-play reinforcement learning pipeline: MCTS search → policy/value network training → evaluation against previous best.

## Architecture

- **Model**: AlphaGoNet — residual CNN with policy + value heads (~300K params for 9×9)
- **Self-play**: Batched MCTS with Dirichlet exploration noise
- **Training**: Policy gradient (cross-entropy) + value regression (MSE), SGD + momentum
- **Eval**: Head-to-head games vs best model, alternating colors

## Colab deployment

### Quick start (fat launch)

```bash
# Build fat launch.py (embeds all source files)
python build_launch.py > launch.py

# Deploy to Colab
colab new --gpu T4 -s alphago-1
colab exec -f launch.py --timeout 120

# Monitor
colab exec -f check_progress.py --timeout 15
```

### Cron watchtower

```bash
bash fetch.sh alphago-1
```

### Config presets

| Config | Games | Sims | Epochs | Purpose |
|--------|-------|------|--------|---------|
| `CONFIG_9X9_FIRST` | 20 | 100 | 3 | Warm-up session (data caching) |
| `CONFIG_9X9_FAST` | 50 | 200 | 5 | Regular training |
| `CONFIG_19X19` | 50 | 800 | 5 | Full board (needs more GPU) |

## Time budget

| Phase | Budget | 540s window |
|-------|--------|-------------|
| Self-play | 40% | 216s |
| Training | 30% | 162s |
| Eval | 20% | 108s |
| Save | 10% | 54s |

**First session:** Use `CONFIG_9X9_FIRST` — reduced self-play (20 games × 100 sims) and minimal eval. Data download + CUDA JIT = 7-10 min overhead makes first session a warmup only. Second session uses cached data.

**Eval is the bottleneck:** Single-tree MCTS at 400 sims/move ≈ 10s per eval game. 100 eval games = ~17 min — won't fit in free-tier GPU window. Reduce `n_eval_games` for colab sessions.

## Output structure

```
/content/alphago-output/
├── logs/train.log       # Timestamped training log
├── metrics.csv          # Per-iteration: loss, win_rate, positions
├── pngs/loss_curves.png # Policy + value loss over iterations
└── summary.json         # Final iteration metadata
```

## Checkpoints (Google Drive)

Mount Drive before training (`colab drivemount`). Checkpoints saved to `/content/drive/MyDrive/alphago-checkpoints/`:

| File | Content |
|------|---------|
| `latest.pt` | Full checkpoint (model + optimizer + iteration) |
| `latest_weights.pt` | Weights only (for inference / network changes) |
| `best.pt` | Best model weights (win rate > 55%) |
| `version.txt` | Best iteration + win rate + ELO estimate |

## Gotchas

See [gotchas.md](gotchas.md) for project-specific issues (NaN/Inf recovery, MCTS eval cost, libomp duplicate warning).
