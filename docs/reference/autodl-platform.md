# AutoDL 算力云 — Platform Reference

AutoDL (autodl.com) is China's largest C-end GPU compute rental platform, operated by 视拓云 (SeetaCloud). Acquired by 东方国信 (300166) in 2025 for majority control (51%). ~30,000 GPUs, 600k+ registered users, serving 6,000+ enterprises and 142 double-first-class universities.

**Relevance to this project:** AutoDL is a potential alternative to Colab/Kaggle for GPU training from China. Unlike Colab's free-tier time limits (~10 min GPU) and Kaggle's 30h/week cap, AutoDL provides paid, stable GPU access with no training-time restrictions — you pay per second and can train for days.

## Quick comparison: Colab vs Kaggle vs AutoDL

| Dimension | Colab (free) | Kaggle | AutoDL |
|-----------|-------------|--------|--------|
| GPU types | T4 (rarely V100/A100) | P100 (30h/wk), T4×2 (30h/wk) | 20+ types: 3090, 4090, A100, H100, Ascend |
| GPU time limit | ~10 min (free GPU), ~12h session max | 30 h/week | No limit (pay per second) |
| Cost | Free | Free | ¥1.5–15/hr depending on GPU |
| Session stability | WebSocket drops 8-12 min (China) | Stable, logs buffer | SSH — no drops, tmux survives |
| Data persistence | Session only (Google Drive mountable) | Session only (dataset + output persist) | System/data disk persist across stops; /root/autodl-fs is durable |
| CLI access | `colab` CLI (WebSocket) | `kaggle` CLI (REST push) | Native SSH + SCP + REST API |
| Proxy required (China) | Yes (complex dual-path) | No (REST via proxy if needed) | No (native SSH via proxy if needed) |
| Multi-account | 4 accounts via $HOME | 4 tokens | Single account, pay-as-you-go |
| Keepalive needed | Yes (complex WebSocket relay) | No | No (SSH + tmux/screen) |
| Best for | Quick experiments, free GPU access | Medium training (≤30h/wk), Kaggle community | Long training, production, reliability |

**Bottom line:** Colab/Kaggle are free but constrained. AutoDL removes the time/reliability constraint but costs money. For projects in this repo that outgrow free-tier limits, AutoDL is the natural paid upgrade path.

## GPU inventory and pricing

Prices are approximate (2025-2026, vary by region: 西北/北京/内蒙/重庆/佛山).

| GPU | VRAM | ~Price/hr | ~Price/month | Use case |
|-----|------|-----------|-------------|----------|
| GTX 1080 Ti | 11 GB | ¥1.0–1.5 | ¥800–1,200 | Code debug, tiny models |
| RTX 3060 | 12 GB | ¥1.5 | ¥1,000–1,500 | Entry-level, light experiments |
| RTX 3090 | 24 GB | ¥3.0–3.5 | ¥2,000–2,500 | CV/NLP training, fine-tuning — sweet spot |
| RTX 4090 | 24 GB | ¥4.0 | ¥2,500–3,500 | Faster 3090, better fp16 perf |
| A100 40GB | 40 GB | ¥15 | ¥6,000–8,000 | LLM training, distributed |
| A100 80GB | 80 GB | ¥18–20 | ¥8,000–10,000 | Large model pretraining |
| H100 | 80 GB | ¥25+ | — | Cutting-edge |
| Ascend 910B | — | Varies | — | 国产算力 (Huawei) |
| MTTS4000 | — | Varies | — | 国产算力 (Moore Threads) |

Billing: per-second, stop-when-shut-down. Also supports daily/weekly/monthly rentals (30-40% discount vs hourly). Membership: 5% discount minimum.

### GPU selection heuristic

- **3090** = default choice for most projects in this repo (RL, CV, small transformers). 24 GB handles MuJoCo, Atari pixels, CIFAR10, IWSLT, nanoGPT comfortably.
- **4090** if the 3090 pool is exhausted or you need fp16 speed.
- **A100 40GB** for LLM fine-tuning (vLLM, text2sql) or multi-GPU distributed.
- **无卡模式 (no-GPU mode)** for environment setup — costs almost nothing.

## Instance lifecycle and storage

### Three storage tiers

| Path | Size | Survives stop? | Survives destroy? | Shared across instances? |
|------|------|---------------|-------------------|------------------------|
| System disk `/` | 30 GB | Yes | **No** | No |
| Data disk `/root/autodl-tmp` | 50 GB+ (expandable) | Yes | **No** | No |
| File storage `/root/autodl-fs` | 20 GB free | **Yes** | **Yes** | Yes (same region) |
| Public data `/root/autodl-pub` | Read-only | — | — | Yes |

**Critical distinction:** Stop (关机) preserves data. Destroy (销毁) wipes everything except `/root/autodl-fs`.

### Workflow pattern

```bash
# 1. Create instance via web console (autodl.com → 容器实例 → 租用新实例)
#    - Select GPU, image (PyTorch + CUDA preinstalled), disk size
#    - Wait 1-3 min for provisioning

# 2. SSH in
ssh -p <port> root@<host>
# e.g. ssh -p 10309 root@connect.nmb1.seetacloud.com

# 3. Verify GPU
nvidia-smi

# 4. Environment (if not using pre-built image)
conda create -n myenv python=3.10
conda activate myenv
pip install torch torchvision

# 5. Upload data (from local machine)
scp -rP <port> ./project/ root@<host>:/root/autodl-tmp/

# 6. Start training in tmux (survives SSH disconnect)
tmux new -s train
python train.py --epochs 500

# 7. Detach: Ctrl+B, D. Reattach: tmux attach -t train

# 8. Download outputs (from local machine)
scp -rP <port> root@<host>:/root/autodl-tmp/output/ ./

# 9. Stop instance when done (via web console or API)
#    Destroy only when you're sure data is backed up
```

### Data upload options

| Method | Speed | Best for |
|--------|-------|----------|
| SCP | Medium | Scripts, small datasets (<1 GB) |
| AutoDL cloud disk (网盘) | Fast | Pre-upload to region, then all instances share |
| HuggingFace/GitHub direct download | Slow | Use `source /etc/network_turbo` academic accelerator |
| FileZilla/XFTP | Medium | GUI users, drag-and-drop |
| tar pipe | Fast | Many small files (tar + ssh stream) |

## Key gotchas

1. **Destroy != Stop.** Destroying an instance deletes ALL data on system and data disks. Only `/root/autodl-fs` survives. Always back up to fs or download before destroying.

2. **System disk is only 30 GB.** pip packages + conda envs + Docker images can fill it fast. Check `df -h` regularly. Put datasets on `/root/autodl-tmp`.

3. **conda solve can hang for hours.** Use mamba: `conda install mamba -c conda-forge && mamba install ...`

4. **External downloads are slow from China.** Use `source /etc/network_turbo` (academic accelerator) for pip/conda/HF. For GitHub, set proxy or use mirrors.

5. **No Docker inside containers.** If you need Docker, rent bare-metal servers (more expensive).

6. **GPU availability fluctuates.** 3090s sell out during peak hours (weekday evenings). Have a backup GPU choice.

7. **No built-in keepalive needed.** SSH + tmux/screen is the standard pattern. Unlike Colab's WebSocket fragility, SSH connections are robust.

8. **Port mapping for custom services.** JupyterLab/Gradio/Streamlit need port mapping configured in instance settings.

9. **Image selection matters.** Pre-installed PyTorch + CUDA images save 15-30 min of setup. Match CUDA version to your PyTorch requirements.

10. **Instance cannot change GPU type.** You must destroy and recreate to switch GPU model. Use `/root/autodl-fs` to persist data across instances.

11. **No free tier.** Unlike Colab/Kaggle, there's no free GPU access. The cheapest GPU (1080 Ti) is ~¥1/hr. For free compute, use Colab/Kaggle first; use AutoDL when you need reliability or long training.

## API and automation

AutoDL has a REST API (Pro API) for programmatic instance management:

- **Docs:** https://www.autodl.com/docs/instance_pro_api/
- **Capabilities:** create/stop/destroy instances, list GPUs, query pricing
- **Auth:** API key from console

There is no official CLI tool comparable to `colab` or `kaggle` CLI. Third-party community projects exist (e.g., `autodl-agent-package` on GitHub) but are not officially supported.

For automated workflows, the practical pattern is:
1. Use the web console to create an instance (GPU selection is interactive anyway)
2. SSH + SCP for everything else (scriptable)
3. Use the API for stop/destroy to avoid forgetting and accruing charges

## Relevance to colab-cli

AutoDL doesn't replace the `colab`/`kaggle` CLI workflows in this repo — it has no equivalent CLI tool. However:

- **When Colab/Kaggle limits are hit** (30h/week cap, 10-min GPU window, WebSocket instability), AutoDL is the fallback.
- **The SSH + tmux + SCP pattern** is simpler and more reliable than Colab's WebSocket + keepalive dance. No need for watchtower crons, relay handoffs, or proxy-path debugging.
- **Training artifacts** (logs/CSVs/PNGs) are easier to stream — just `scp` periodically or use `tail -f` over SSH.
- **Cost is the tradeoff.** A 3090 at ¥3/hr × 100h = ¥300. Same compute on Colab free = ¥0 but requires ~10 sessions with relay handoffs.

### When to use which

| Scenario | Platform |
|----------|----------|
| Quick debug, <8 min training | Colab (free, T4) |
| Medium training, ≤30h/week, P100 ok | Kaggle (free) |
| Long training, need stability | AutoDL (paid) |
| Multi-GPU or A100/H100 | AutoDL (paid) |
| LLM inference (vLLM) | Colab T4 for small models; AutoDL A100 for 7B+ |
| Hyperparameter sweeps | Kaggle (free P100) or AutoDL (paid, faster) |
| Production / deadline-sensitive | AutoDL (paid, reliable) |

## Sources

- AutoDL official docs: https://api.autodl.com/docs/
- Pro API: https://www.autodl.com/docs/instance_pro_api/
- SSH guide: https://api.autodl.com/docs/ssh/
- Data upload: https://api.autodl.com/docs/scp
- NAS/file storage: https://api.autodl.com/docs/nas/
- 东方国信 acquisition report (信达证券, 2025-10): https://data.eastmoney.com/report/info/AP202510091758934043.html
