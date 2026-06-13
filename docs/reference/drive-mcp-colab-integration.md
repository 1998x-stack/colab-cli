# Google Drive MCP 与 colab-cli 集成分析报告

> **背景**: 用户从中国使用 `colab` CLI (v0.5.9) 运行 Google Colab，采用多账户设置（4 个 Google 账户）。
> 所有 `/content/` 文件在会话结束后丢失。当前 checkpoint 只保存到 VM 本地，缺乏持久化方案。
>
> **撰写日期**: 2026-06-11

---

## 目录

1. [Google Drive MCP 生态](#1-google-drive-mcp-生态)
2. [Colab CLI Drive 集成原理解析](#2-colab-cli-drive-集成原理解析)
3. [Checkpoint 持久化工作流设计](#3-checkpoint-持久化工作流设计)
4. [多账户 Drive 管理](#4-多账户-drive-管理)
5. [替代存储方案对比](#5-替代存储方案对比)
6. [实现建议与优先级](#6-实现建议与优先级)

---

## 1. Google Drive MCP 生态

### 1.1 核心对比

| 项目 | 语言 | 读写 | Sheets | 认证方式 | 成熟度 | 最后更新 |
|------|------|------|--------|----------|--------|----------|
| `@modelcontextprotocol/server-gdrive` | TypeScript | 只读 | 否 | OAuth2 | 官方, 稳定 | 2025-01 |
| `dennisonbertram/mcp-gdrive` | TypeScript | 读写 | 是 | OAuth2 / Service Account | 功能最全 | 2025-07 |
| `ankitpyc/gdrive-mcp-server` | Go | 读写 | 否 | Service Account | 基础功能 | 2025-11 |
| `rishipradeep-think41/google-drive-mcp` | Node.js | 读写 | 否 | OAuth2 | 中等 | 2024-2025 |
| `@chieflatif/google-mcp` | TypeScript | 读写 | 是 | OAuth2 | 生产就绪 | 2025 |
| `terra-mcp-google` | TypeScript | 读写 | 是 | OAuth2+PKCE | 中等 | 2025 |

### 1.2 关键 MCP 服务器详解

#### Anthropic 官方: `@modelcontextprotocol/server-gdrive`

- **地址**: npm 包，约 8K 周下载量
- **能力**: search（搜索文件）、根据 file ID 读取文件内容（`gdrive:///<file_id>`）
- **自动格式转换**: Google Docs → Markdown, Sheets → CSV, Slides → Text, Drawings → PNG
- **认证**: OAuth2 Desktop 流，凭证文件保存在本地
- **局限**: **只读**（`drive.readonly` scope），不支持上传、下载二进制、权限管理
- **适用场景**: Claude 读取 Drive 文档辅助对话，**不适合 checkpoint 上传**

#### 功能最全: `dennisonbertram/mcp-gdrive`

- **地址**: [github.com/dennisonbertram/mcp-gdrive](https://github.com/dennisonbertram/mcp-gdrive)
- **核心能力**:
  - 文件 CRUD（创建、读取、更新、删除）
  - 文件夹管理（创建层级、移动、获取目录树）
  - 权限控制（共享、公开链接 + 过期时间）
  - Google Sheets 集成（创建、读写单元格、批量操作、格式化）
  - 交互式 Prompt（按模式整理文件、备份、归档、清理重复）
- **工具数**: 20+，覆盖 Drive + Sheets
- **认证**: 支持 OAuth2 和 Service Account 两种模式
- **适用场景**: 需要完整的文件读写 + 权限管理，是最理想的 Drive MCP server 候选

#### 工具最丰富: `@chieflatif/google-mcp`

- **地址**: npm，28 个工具覆盖 Gmail/Calendar/Sheets/Docs/Drive
- **Drive 专属工具**: `drive_list`, `drive_search`, `drive_download`, `drive_upload`, `drive_share`, `drive_create_folder`
- **上传限制**: 最大 10MB（对 checkpoint 文件偏小，大模型权重需要分片）
- **认证**: OAuth2 + 自动 token 刷新
- **适用场景**: 轻量文件操作 + 多 Google 服务协同

### 1.3 认证模型详解

#### OAuth2 Desktop Flow（最常用）

```
用户运行 MCP server
  → 打开浏览器弹出 Google 登录页
  → 用户授权 scope
  → 获取 refresh_token + access_token
  → 保存到本地文件 (~/.config/gdrive-xxx/token.json)
  → 后续自动 refresh（除非 refresh_token 过期）
```

**注意**: OAuth consent screen 设为 "Testing" 模式时，refresh_token 7 天过期。
**解决方案**: 发布到 "Production" 模式或使用 Service Account。

#### Service Account

- 不需要浏览器交互
- 需要在 Google Cloud 创建 Service Account 并授予 Drive 权限
- 适合 CI/CD 和自动化场景
- **局限**: Service Account 拥有自己的 Drive，不能直接访问个人用户的 Drive（除非用户主动共享文件夹）

#### 与 Colab OAuth Token 共享的可能性

colab-cli 的 `auth.py` 使用的 scope 包括：

```python
PUBLIC_SCOPES = [
    "openid",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/colaboratory",
    "https://www.googleapis.com/auth/drive.file",  # ← Drive scope
]
```

关键发现：colab-cli 的 OAuth token **已经包含 `drive.file` scope**，这意味着：

- **可以复用同一个 token 来访问 Drive API**
- 但 `drive.file` scope 只能访问应用自己创建或用户通过此应用打开的文件
- 要访问用户的完整 Drive，需要使用 `drive` 或 `drive.readonly` scope
- 目前的 token 存储在 `~/.config/colab-cli/token.json`

**结论**: 理论上可以通过扩展 scope 来让 colab-cli 同时获得 Drive 访问权限。
但 MCP server 是独立进程，需要独立管理 token，共享有一定复杂度。

---

## 2. Colab CLI Drive 集成原理解析

### 2.1 `colab drivemount` 实现流程

源码位置: `colab_cli/commands/automation.py` (192-211 行)

```
colab drivemount -s training [/content/drive]
  ↓
发送代码到 VM Jupyter kernel:
  from google.colab import drive
  drive.mount('/content/drive')
  ↓
VM 上的 google.colab.drive.mount() 执行:
  1. 检查是否已挂载
  2. 如未挂载，向 Jupyter kernel 发送 dfs_ephemeral auth 请求
  3. 本地 colab CLI 拦截 colab_request（drivefs_hook 钩子）
  4. 通过 OAuth2 token 向 credentials-propagation API 发送请求
  5. 如果 token 无权，返回 unauthorized_redirect_uri
  6. 用户浏览器打开该 URI 完成 Drive 授权
  7. VM 挂载 Drive 到 /content/drive/
```

### 2.2 核心机制: `drivefs_hook`

代码位于 `automation.py` 的 51-128 行。关键步骤：

1. **拦截**: 当 Jupyter kernel 发出 `colab_request` 且 `authType == "dfs_ephemeral"` 时触发
2. **传播凭证**: 用本地 OAuth token 向 `{colab_domain}/tun/m/credentials-propagation/{endpoint}` 发送请求
3. **静默授权**: 如果 token 有权限，自动完成 Drive 授权（无需用户交互）
4. **失败回退**: 如果 token 无权，打印 URI 等待用户手动授权

### 2.3 文件操作命令

colab CLI 支持以下文件操作：

| 命令 | 功能 | 实现 |
|------|------|------|
| `colab upload LOCAL REMOTE` | 上传到 VM | Contents API |
| `colab download REMOTE LOCAL` | 从 VM 下载 | Contents API |
| `colab ls [PATH]` | 列出 VM 文件 | Contents API |
| `colab rm PATH` | 删除 VM 文件 | Contents API |
| `colab edit PATH` | 编辑 VM 文件 | 下载→本地编辑→上传 |

### 2.4 当前架构的局限性

```
[VM /content/] → → → 会话死亡，文件丢失
       ↓
  colab download
       ↓
[本地机器]      → → → 手动操作，容易遗漏
```

- Checkpoint 只能手动下载，没有自动化机制
- Multi-account 场景下需要分别操作
- 代理/网络不稳定时下载可能中断
- 当前 `checkpoint.py` 只保存到 `/content/checkpoints/`，无 Drive 写入

---

## 3. Checkpoint 持久化工作流设计

### 3.1 方案总览

有三种根本不同的架构路径：

| 方案 | 实时性 | 复杂度 | 可靠性 | 需要外部服务 |
|------|--------|--------|--------|-------------|
| **A**: VM 直写 Drive | 实时 | 低 | 中 | 无 |
| **B**: 本地拉取 + 上传 | 异步 | 中 | 高 | MCP/脚本 |
| **C**: Chat 感知的自动化 | 按需 | 高 | 最高 | Claude + MCP |

### 3.2 方案 A: VM 直写 Google Drive（推荐首选）

在训练脚本中直接挂载 Drive 并写入，是目前最成熟、最可靠的方式。

```
训练循环
  ↓ 每 N 个 epoch
save_checkpoint("/content/drive/MyDrive/checkpoints/project/epoch_N.pt")
  ↓
文件直接写入 Drive 云端
  ↓
会话死亡 → checkpoint 安全在 Drive 上
  ↓
新会话启动 → mount Drive → 从 Drive 加载 checkpoint 恢复训练
```

#### 实施步骤

**步骤 1: 修改 `launch.py`**，在训练前挂载 Drive

```python
# 在 launch.py 的依赖安装之后加入
print("[launch] Mounting Google Drive for checkpoint persistence...")
from google.colab import drive
drive.mount('/content/drive')
```

**步骤 2: 修改 `checkpoint.py`**，支持 Drive 路径

```python
import os

# 检测 Drive 是否挂载
def get_checkpoint_dir(project: str = "transformer_iwslt") -> str:
    """优先使用 Drive，回退到本地 /content/checkpoints/"""
    drive_path = f"/content/drive/MyDrive/colab-checkpoints/{project}"
    local_path = "/content/checkpoints"
    if os.path.exists("/content/drive"):
        os.makedirs(drive_path, exist_ok=True)
        return drive_path
    os.makedirs(local_path, exist_ok=True)
    return local_path


def save_checkpoint_drive(
    model, optimizer, scheduler, epoch, metrics, config,
    project: str = "transformer_iwslt",
):
    """保存 checkpoint 到 Drive（若已挂载）"""
    ckpt_dir = get_checkpoint_dir(project)
    path = os.path.join(ckpt_dir, f"checkpoint_epoch{epoch:02d}.pt")
    save_checkpoint(path, model, optimizer, scheduler, epoch, **metrics, config=config)
    print(f"[checkpoint] Saved to {path}")

    # 同时写入 latest 指针文件（便于恢复时定位最新 checkpoint）
    latest_path = os.path.join(ckpt_dir, "latest.txt")
    with open(latest_path, "w") as f:
        f.write(path)
    print(f"[checkpoint] Updated latest pointer: {latest_path}")


def load_latest_checkpoint(project: str, model, device):
    """从 Drive 加载最新的 checkpoint"""
    ckpt_dir = get_checkpoint_dir(project)
    latest_path = os.path.join(ckpt_dir, "latest.txt")
    if os.path.exists(latest_path):
        with open(latest_path) as f:
            ckpt_path = f.read().strip()
        if os.path.exists(ckpt_path):
            return load_checkpoint(ckpt_path, model, device)
    # 回退：搜索最新 epoch 的 checkpoint
    import glob
    ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "checkpoint_epoch*.pt")))
    if ckpts:
        return load_checkpoint(ckpts[-1], model, device)
    return None  # 无 checkpoint，从头开始
```

**步骤 3: 在 `train.py` 训练循环中整合**

```python
# 训练前尝试恢复
result = load_latest_checkpoint("transformer_iwslt", model, device)
if result:
    opt_state, sched_state, start_epoch, metrics, config = result
    optimizer.load_state_dict(opt_state)
    if scheduler and sched_state:
        scheduler.load_state_dict(sched_state)
    print(f"[train] Resumed from epoch {start_epoch}")
else:
    start_epoch = 0

# 训练循环中保存
for epoch in range(start_epoch, total_epochs):
    train_one_epoch(...)
    if epoch % save_interval == 0:
        save_checkpoint_drive(model, optimizer, scheduler, epoch, metrics, config)
```

#### 方案 A 的优缺点

**优点**:
- 零额外架构，完全基于 Colab 原生能力
- 实时写入，epoch 完成即持久化
- 不依赖本地网络（VM 到 Drive 走 Google 内部网络，速度快）
- 多账户天然支持（每个账户挂载自己的 Drive）

**缺点**:
- Drive I/O 延迟较高（挂载是 FUSE 文件系统）
- 大文件写入可能在会话死亡时损坏
- 每日免费账户的 Drive 存储上限（15GB）

#### 优化: 写入策略

```python
def save_checkpoint_safe(path, model, optimizer, scheduler, epoch, metrics, config, config_dict):
    """先写本地临时文件，再复制到 Drive，降低损坏风险"""
    local_tmp = f"/content/tmp_checkpoint_{epoch}.pt"
    # 1. 写本地（快）
    torch.save({...}, local_tmp)
    # 2. 复制到 Drive
    if os.path.exists("/content/drive"):
        import shutil
        shutil.copy2(local_tmp, path)
        print(f"[checkpoint] Copied to Drive: {path}")
    # 3. 可选：删除本地临时文件
    # os.remove(local_tmp)
```

### 3.3 方案 B: 本地拉取 + Drive 上传（MCP 辅助）

利用 MCP server 从本地机器将已下载的 checkpoint 上传到 Drive。

```
VM 训练（checkpoint 在 /content/ 本地）
  ↓
colab download   ← cron job 定时拉取
  ↓
本地 checkpoint 文件
  ↓
Drive MCP Server  ← 自动上传到 Drive
  ↓
Drive 云端存储
```

#### 实施步骤

**步骤 1: 安装 Drive MCP Server**

```bash
# 安装 dennisonbertram/mcp-gdrive（功能最全）
npx -y @modelcontextprotocol/gdrive-mcp
```

**步骤 2: 配置 Claude Desktop / Claude CLI MCP**

```json
{
  "mcpServers": {
    "gdrive": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/gdrive-mcp"],
      "env": {
        "GDRIVE_CLIENT_ID": "your_client_id",
        "GDRIVE_CLIENT_SECRET": "your_client_secret",
        "GDRIVE_REDIRECT_URI": "http://localhost:3000/oauth2callback",
        "GDRIVE_CREDS_DIR": "~/.config/gdrive-mcp"
      }
    }
  }
}
```

**步骤 3: 自动化上传脚本**

```python
#!/usr/bin/env python3
"""checkpoint_sync.py: 同步本地 checkpoint 到 Google Drive。

用法:
  python checkpoint_sync.py --project transformer_iwslt --dry-run
  python checkpoint_sync.py --project transformer_iwslt --upload
"""
import argparse
import json
import os
import glob
import hashlib
from pathlib import Path

# Drive MCP server 的 REST 接口（假设运行在本地）
MCP_SERVER_URL = "http://localhost:3100"  # 需根据实际 MCP server 配置调整

REMOTE_BASE = "colab-checkpoints"


def get_local_checkpoints(project: str, base_dir: str) -> list:
    """扫描本地已下载的 checkpoint 文件"""
    pattern = os.path.join(base_dir, project, "checkpoint_epoch*.pt")
    # 也支持最新指针文件中的路径
    latest_path = os.path.join(base_dir, project, "latest.txt")
    ckpts = sorted(glob.glob(pattern))
    return ckpts


def get_drive_file_list(project: str):
    """通过 MCP 工具获取 Drive 上已有文件列表"""
    # 模拟 MCP tool call: gdrive_list_files
    # 实际实现需根据所选 MCP server 的协议调整
    pass


def sync_checkpoints(project: str, base_dir: str, dry_run: bool = False):
    """将本地 checkpoint 同步到 Drive（增量上传）"""
    ckpts = get_local_checkpoints(project, base_dir)
    if not ckpts:
        print(f"[sync] No local checkpoints found for {project}")
        return

    for ckpt_path in ckpts:
        fname = os.path.basename(ckpt_path)
        remote_path = f"{REMOTE_BASE}/{project}/{fname}"
        if dry_run:
            print(f"[sync] Would upload: {ckpt_path} → {remote_path}")
        else:
            # 实际调用 MCP upload 工具
            print(f"[sync] Uploading: {ckpt_path} → {remote_path}")
            # upload_file_via_mcp(ckpt_path, remote_path)
    print(f"[sync] Done. {len(ckpts)} files {'would be ' if dry_run else ''}synced.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--base-dir", default="projects")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    sync_checkpoints(args.project, args.base_dir, args.dry_run)
```

#### 方案 B 的优缺点

**优点**:
- 不修改 VM 上已有的训练流程
- 可以用 MCP 实现增量上传和校验
- MCP server 可被 Claude 会话直接调用

**缺点**:
- **中国网络环境**: 从中国上传到 Google Drive 可能慢或不稳定（需代理）
- 复杂度过高：需要管理本地→VM 下载 + 本地→Drive 上传两个阶段
- 时效性差：下载时机难以对齐训练完成时刻
- MCP server 需要保持运行，增加管理负担

### 3.4 方案 C: Chat 感知的自动化（推荐次选）

利用 Claude 会话 + MCP server 感知训练状态，自动调度同步。

```
Claude 会话
  ↓ 感知到训练完成 / 新 checkpoint 出现
通过 colab CLI 下载
  ↓
通过 MCP 上传到 Drive
  ↓
记录到本地状态文件
```

这个方案不需要额外脚本，完全在 Claude Code 会话中完成。在 CLAUDE.md 中加入指令即可。

#### CLAUDE.md 配置

```markdown
## Checkpoint 管理（Drive MCP）

训练完成后，自动执行：

1. `colab download <session> /content/checkpoints/checkpoint_epoch*.pt projects/<project>/output/`
2. 通过 `gdrive_upload_file` MCP 工具将 checkpoint 上传到 Drive:
   - 路径: `colab-checkpoints/<project>/`
   - 同时更新 `latest.txt` 指针
   - 清理旧 checkpoint（保留最近 3 个）
```

#### 方案 C 的优缺点

**优点**:
- 零基础设施，完全对话驱动
- 灵活：Claude 可智能判断何时需要同步
- 可组合：下载 + 上传 + 清理一次完成

**缺点**:
- 需要人工监督（Claude Code 会话）
- 不能实时同步（只对话时才触发）
- 代理不稳定时上传可能失败需重试

---

## 4. 多账户 Drive 管理

### 4.1 账户映射策略

用户有 4 个 Google 账户（colab, cc, cb, clb），每个账户关联一个独立的 Drive。

建议的目录结构：

```
<每个账户的 Drive>/My Drive/
  └── colab-checkpoints/
      ├── transformer_iwslt/          ← 统一项目名
      │   ├── checkpoint_epoch01.pt
      │   ├── checkpoint_epoch02.pt
      │   ├── latest.txt              ← 指向最新的 checkpoint
      │   └── training_config.json    ← 实验配置快照
      ├── alexnet_imagenette/
      │   └── ...
      └── experiments.csv             ← 实验汇总表格（可选）
```

### 4.2 账户管理建议

| 账户 | 用途 | Drive 存储内容 |
|------|------|---------------|
| colab (hackxie1998) | 主要实验 | 主项目 checkpoint |
| cc (xbetterdetermine) | 并行实验 | 副项目 checkpoint |
| cb (stefaniehu929) | 备用 | 存档/备份 |
| clb (xieminghack) | 备用 | 存档/备份 |

### 4.3 跨账户 Checkpoint 恢复

当需要在不同账户间切换时：

```python
# launch.py 中从 Drive 恢复 checkpoint
import os

# 根据当前 VM 的账户，读取对应 Drive 的 checkpoint
drive_base = "/content/drive/MyDrive/colab-checkpoints"

def find_latest_checkpoint(project: str) -> str | None:
    """在所有可能的 Drive 路径中查找最新 checkpoint"""
    # 优先走 latest.txt 指针
    for root, dirs, files in os.walk(drive_base):
        if "latest.txt" in files:
            with open(os.path.join(root, "latest.txt")) as f:
                path = f.read().strip()
                if os.path.exists(path):
                    return path

    # 回退：找文件名中 epoch 最大的
    import glob, re
    ckpts = glob.glob(f"{drive_base}/{project}/checkpoint_epoch*.pt")
    if ckpts:
        # 提取 epoch 号排序
        def epoch_key(p):
            m = re.search(r"epoch(\d+)", p)
            return int(m.group(1)) if m else 0
        return max(ckpts, key=epoch_key)

    return None
```

### 4.4 Drive MCP 多账户配置

如果使用 `google-mcp-suite`（支持多账户），可以为每个账户启动一个 MCP server 实例：

```bash
# 为 colab 账户启动 MCP
GOOGLE_MCP_ACCOUNT=colab npx google-mcp-suite drive

# 为 cc 账户启动 MCP
GOOGLE_MCP_ACCOUNT=cc npx google-mcp-suite drive
```

如果使用 `dennisonbertram/mcp-gdrive`，不同账户的 token 文件存储在不同路径：

```bash
# 账户 1
GDRIVE_CREDS_DIR=~/.config/gdrive-mcp-colab npx @modelcontextprotocol/gdrive-mcp

# 账户 2
GDRIVE_CREDS_DIR=~/.config/gdrive-mcp-cc npx @modelcontextprotocol/gdrive-mcp
```

---

## 5. 替代存储方案对比

### 5.1 综合对比

| 方案 | 免费额度 | 中国访问 | 上传速度 | 集成复杂度 | 适合场景 |
|------|---------|----------|---------|-----------|---------|
| **Google Drive (Mount)** | 15GB/账户 | 差（需代理） | 中 | 极低 | 首选，实时保存 |
| **Hugging Face Hub** | 无限模型存储 | 好 | 快 | 低 | 模型发布/分享 |
| **Google Cloud Storage** | 无免费层 | 差 | 快(内网) | 中 | 大型实验 |
| **rclone + Drive** | 15GB/账户 | 差 | 中 | 中 | 批量迁移 |
| **本地文件** | 无限 | 最好 | N/A | 低 | 快速访问 |

### 5.2 Hugging Face Hub

- **优势**: 在国内访问速度快（HF 镜像 `hf-mirror.com`），免费存储无限
- **集成**: PyTorch 原生支持 `push_to_hub`
- **示例**: `model.push_to_hub("username/project-checkpoint")`

```python
# 从 Colab 上传 checkpoint 到 HF Hub
from huggingface_hub import HfApi, create_repo

api = HfApi()
repo_id = f"{hf_username}/{project_name}-checkpoints"

# 确保 repo 存在
try:
    create_repo(repo_id, exist_ok=True)
except:
    pass

# 上传 checkpoint
api.upload_file(
    path_or_fileobj=f"/content/checkpoints/checkpoint_epoch{epoch:02d}.pt",
    path_in_repo=f"checkpoint_epoch{epoch:02d}.pt",
    repo_id=repo_id,
)
```

### 5.3 Google Cloud Storage (GCS)

- **优势**: 可以从 Colab VM 直接通过内网访问，速度快
- **劣势**: 需要开通 GCP 账号和 billing，无免费额度
- **认证**: 复用 colab-cli 的 OAuth token（已有 `cloud-platform` scope）

```python
# 从 Colab 上传到 GCS
from google.colab import auth
auth.authenticate_user()

from google.cloud import storage
client = storage.Client()
bucket = client.bucket("colab-checkpoints")
blob = bucket.blob(f"transformer_iwslt/checkpoint_epoch{epoch:02d}.pt")
blob.upload_from_filename(f"/content/checkpoints/checkpoint_epoch{epoch:02d}.pt")
```

### 5.4 rclone + Google Drive

rclone 是从 Colab 同步文件到 Drive 的成熟方案，特别是大文件和批量操作。

**在 Colab VM 上使用**:

```python
# 安装 rclone
!curl https://rclone.org/install.sh | sudo bash

# 使用预配置的 rclone.conf（从 Drive 挂载点读取）
!rclone copy /content/checkpoints/ gdrive:colab-checkpoints/transformer_iwslt/ \
    --progress \
    --drive-chunk-size=64M \
    --checksum
```

**优势**: 支持断点续传、校验和验证、带宽控制
**劣势**: 需要额外安装和配置，中国网络问题同样存在

### 5.5 推荐策略

**最佳组合**:

1. **主方案**: VM 直写 Drive（方案 A）— 零依赖，实时持久化
2. **辅助方案**: Drive → HF Hub 备份（通过 MCP 或脚本）— 跨平台可访问
3. **紧急方案**: `colab download` 本地备份 — 网络最可靠

```
VM 训练 → checkpoint 写 Drive（实时）→ 会话死亡 → Drive 保留
                          ↓（可选）
                   HF Hub 上传（跨平台备份）
                          ↓（紧急）
                  colab download 本地（离线存档）
```

---

## 6. 实现建议与优先级

### 6.1 P0: VM 直写 Drive（立即实施，预计 1-2 小时）

这是最可靠、最符合现有架构的方案。

**TODOs**:
1. 为每个项目添加 Drive checkpoint 保存函数
2. 修改 `launch.py` 在训练前自动挂载 Drive
3. 修改 `train.py` 从 Drive 加载最新 checkpoint 恢复训练
4. 更新 `checkpoint.py` 的路径检测逻辑

**已完成的项目需要修改**:
- `projects/transformer_iwslt/checkpoint.py` — 添加 Drive 路径支持
- `projects/transformer_iwslt/launch.py` — 添加自动挂载
- `projects/transformer_iwslt/train.py` — 添加 Drive 恢复逻辑
- `projects/alexnet_imagenette/train.py` — 同理

### 6.2 P1: Drive MCP 集成（本周内）

**TODOs**:
1. 选择一个 MCP server（建议 `dennisonbertram/mcp-gdrive`）
2. 在本地完成 OAuth 配置并验证上传/下载功能
3. 编写 `checkpoint_sync.py` 脚本，封装 MCP 工具调用
4. 在 CLAUDE.md 中添加 Drive MCP 说明

**需注意**:
- 中国网络访问 Google Drive 需要代理
- 上传大文件（>10MB）可能需要分片或使用 rclone
- MCP server 的 OAuth 配置需要 Google Cloud Console 操作

### 6.3 P2: 多账户管理策略

**TODOs**:
1. 为每个账户建立独立的 Drive 目录结构
2. 编写跨账户 checkpoint 聚合查询工具
3. 实验完成自动标记和归档（Drive ⇔ 本地双向同步）

### 6.4 P3: 自动化同步流水线

**TODOs**:
1. cron job + colab CLI 定时拉取 checkpoint
2. MCP 自动上传到 Drive
3. 增量同步和旧 checkpoint 清理
4. 训练完成通知（可选）

### 6.5 快速开始的命令集

```bash
# 1. 启动训练并挂载 Drive
colab new --gpu T4 -s training
colab upload *.py /content/
colab drivemount -s training              # 挂载 Drive
colab exec -s training -f launch.py --timeout 120

# 2. 验证 checkpoint 已写入 Drive
colab exec -s training -c "
import os
path = '/content/drive/MyDrive/colab-checkpoints/transformer_iwslt/'
for f in os.listdir(path):
    print(f, os.path.getsize(f))
"

# 3. 新会话恢复训练
colab exec -s training -f launch.py --timeout 120  # launch.py 自动从 Drive 恢复

# 4. 如需 MCP 辅助上传（从本地）
export GDRIVE_CREDS_DIR=~/.config/gdrive-mcp-colab
npx -y @modelcontextprotocol/gdrive-mcp
# 在 Claude 中: "上传 projects/transformer_iwslt/output/ 下的 checkpoint 到 Drive"
```

### 6.6 CLAUDE.md 更新建议

将以下内容添加到项目根目录 `/Users/mx/Desktop/projects/colab-cli/CLAUDE.md` 中：

```markdown
## Drive MCP 集成

- `colab drivemount -s <session>` 在 VM 上挂载 Google Drive
- 训练脚本自动保存 checkpoint 到 Drive: `/content/drive/MyDrive/colab-checkpoints/<project>/`
- `checkpoint.py` 自动检测 Drive 挂载状态，优先使用 Drive 路径
- 恢复训练时自动从 Drive 加载最新 checkpoint
- 本地可通过 Drive MCP server 管理已同步的 checkpoint
```

---

## 附录: 关键源码分析

### colab-cli OAuth 流程 (auth.py)

```
1. 检查 ~/.config/colab-cli/token.json 是否存在
2. 若存在且有效 → 直接使用
3. 若过期且有 refresh_token → 自动 refresh
4. 若无 → 启动浏览器 OAuth 流程 (InstalledAppFlow)
5. 保存 token.json 供后续使用
```

### drivemount 流程 (automation.py)

```
1. 发送 `from google.colab import drive; drive.mount(path)` 到 kernel
2. kernel 执行 mount，触发 dfs_ephemeral auth 请求
3. drivefs_hook 拦截请求，用 OAuth token 向 credentials-propagation API 传播凭证
4. 授权成功 → VM 挂载 Drive
5. 授权失败 → 打印 URI，等待用户手动完成
```

### Token Scope 对比

| Scope | colab-cli | MCP-gdrive |
|-------|-----------|------------|
| `drive.file` | 是 | 否 |
| `drive.readonly` | 否 | 是（官方 MCP）|
| `drive` | 否 | 是（功能全的 MCP）|
| `colaboratory` | 是 | 否 |
| `cloud-platform` | 是 | 否 |

---

## 参考资料

- [Google Colab CLI README](https://github.com/google/colab-cli) — colab-cli 官方文档
- [@modelcontextprotocol/server-gdrive (npm)](https://www.npmjs.com/package/@modelcontextprotocol/server-gdrive) — 官方 Drive MCP 服务器
- [dennisonbertram/mcp-gdrive (GitHub)](https://github.com/dennisonbertram/mcp-gdrive) — 功能最全的 Drive MCP 服务器
- [@chieflatif/google-mcp (npm)](https://www.npmjs.com/package/@chieflatif/google-mcp) — 多服务 MCP 套件
- [rclone 官方文档](https://rclone.org/drive/) — rclone + Google Drive 配置
- [Google Drive API: Resumable Upload](https://developers.google.com/workspace/drive/api/guides/manage-uploads) — 大文件断点续传
- [google-mcp-suite (npm)](https://www.npmjs.com/package/google-mcp-suite) — 多账户 MCP 套件
- [mcp-gdrive-fixed (npm)](https://www.npmjs.com/package/mcp-gdrive-fixed) — 修复了 stdout/stderr 问题的版本
