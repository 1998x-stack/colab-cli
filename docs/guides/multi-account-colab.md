# Colab CLI 多账户使用指南

## 当前机器已配置账户

| Alias | Email | HOME |
|-------|-------|------|
| `colab` | hackxie1998@gmail.com | default `~` |
| `cb` | stefaniehu929@gmail.com | `~/colab-accounts/account-b` |
| `cc` | xbetterdetermine@gmail.com | `~/colab-accounts/account-c` |
| `clb` | xieminghack@gmail.com | `~/colab-accounts/account-clb` |
| `clab` | xieminghacker@gmail.com | `~/colab-accounts/account-clab` |

Alias 定义在 `~/.zshrc` 中，proxy env vars 已内置。

## 结论

**Colab CLI (v0.5.9) 不原生支持多账户 / 多 profile 切换。** 但可以通过变通方案实现。

## 源码级认证架构分析

通过阅读 `colab_cli` 全部 20 个源文件（`auth.py`, `cli.py`, `common.py`, `state.py`, `client.py` 等），完整梳理了认证链：

### 文件路径布局

```
~/.colab-cli-oauth-config.json          ← OAuth client 配置 (可通过 -c 自定义)
~/.config/colab-cli/
├── token.json                          ← OAuth2 缓存 token (硬编码，无法自定义)
├── sessions.json                       ← 会话状态 (可通过 --config 自定义)
├── settings.json                       ← 全局设置 (硬编码)
├── colab.log                           ← 调试日志 (硬编码)
└── history/                            ← 会话历史 (硬编码)
    └── <session-name>.jsonl
```

### 关键源码证据

**1. token.json 硬编码 — `auth.py:54`**

```python
TOKEN_CONFIG_PATH = os.path.expanduser("~/.config/colab-cli/token.json")
```

这是整个 CLI 中唯一的 token 存储路径，不接收任何参数，不受任何 CLI flag 或环境变量影响。OAuth2 认证后总是写入此路径，启动时总是从此路径读取。

**2. sessions.json 可自定义 — `state.py:108-110`**

```python
class StateStore(_LockedFileStore):
    def __init__(self, path: Optional[str] = None):
        if not path:
            path = os.path.expanduser("~/.config/colab-cli/sessions.json")
```

通过 `--config` flag → `common.py:45` 传入自定义路径。但 **仅影响 session 记录**，不影响 token、settings、history。

**3. settings.json 硬编码 — `common.py:49-53`**

```python
@property
def settings_store(self):
    if self._settings_store is None:
        self._settings_store = SettingsStore()  # 总是默认路径
```

即使用了 `--config`，settings 仍指向 `~/.config/colab-cli/settings.json`。

**4. history 目录硬编码 — `history.py:22`**

```python
class HistoryLogger:
    def __init__(self, log_dir: str = "~/.config/colab-cli/history"):
```

`common.py` 中创建 HistoryLogger 不传参数，使用默认路径。

**5. keep-alive 守护进程继承 auth 和 config — `session.py:398-402`**

```python
cmd = [sys.executable, "-m", "colab_cli.cli"]
if auth_provider is not None:
    cmd.append(f"--auth={auth_provider.value}")
if config_path is not None:
    cmd.extend(["--config", config_path])
```

守护进程会正确继承父进程的 `--auth` 和 `--config`，保证一致性。

**6. 无环境变量支持**: 全局搜索 `environ`、`getenv`、`COLAB` — 零结果。路径完全依赖 `os.path.expanduser("~/...")` 和 CLI flags。

### 认证模式

| 模式 | 凭据来源 | 可配置项 |
|------|---------|---------|
| OAuth2 (默认) | `-c` 指定的 OAuth config → `token.json` | `-c` 可自定义 OAuth config 路径 |
| ADC | `gcloud auth application-default login` 或 `GOOGLE_APPLICATION_CREDENTIALS` | 通过 gcloud config / 环境变量 |

两种模式使用**完全独立的凭据存储路径**，可以同时使用。

## 多账户方案

### 方案一：独立 HOME 目录（推荐，完全隔离）

利用所有路径都从 `$HOME` 派生的特性，实现彻底隔离：

```bash
# 设置目录结构
mkdir -p ~/colab-accounts/account-a/.config/colab-cli
mkdir -p ~/colab-accounts/account-b/.config/colab-cli

# 账户 A：首次认证
HOME=~/colab-accounts/account-a colab new --gpu T4 -s training-a

# 账户 B：首次认证
HOME=~/colab-accounts/account-b colab new --gpu T4 -s training-b

# 后续使用（建议设为 alias）
alias colab-a='HOME=~/colab-accounts/account-a colab'
alias colab-b='HOME=~/colab-accounts/account-b colab'

colab-a sessions
colab-b sessions
colab-a exec -s training-a -f train.py &
colab-b exec -s training-b -f train.py &   # 可同时运行
```

**隔离范围**：token.json, sessions.json, settings.json, colab.log, history/ — **全部隔离**

**优点**：完美隔离，可同时操作，互不干扰
**缺点**：需要分别完成 OAuth 认证（各弹一次浏览器）；占用额外磁盘空间（每个 ~10KB）

### 方案二：混合认证模式 OAuth2 + ADC（可同时使用）

OAuth2 和 ADC 使用不同的凭据存储，天然互不冲突：

```bash
# 账户 A：OAuth2 模式（默认）
colab --auth oauth2 new --gpu T4 -s training-a

# 账户 B：ADC 模式
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,\
https://www.googleapis.com/auth/userinfo.email,\
https://www.googleapis.com/auth/colaboratory

colab --auth adc new --gpu T4 -s training-b
```

**注意**：sessions.json、settings.json、colab.log 共享，用不同 session 名区分即可。

**优点**：零额外配置，可真正同时操作
**缺点**：一个账户必须用 ADC；日志和 session 记录混在一起（可通过 `--config` 给 ADC 账户指定单独的 sessions.json 缓解）

### 方案三：Shell token 切换（轮流使用，不可同时）

```bash
# 认证账户 A 并存档
colab new --gpu T4 -s training-a  # 完成 OAuth
cp ~/.config/colab-cli/token.json ~/.config/colab-cli/token-a.json

# 认证账户 B 并存档
mv ~/.config/colab-cli/token.json ~/.config/colab-cli/token-a.bak
colab new --gpu T4 -s training-b  # 完成 OAuth
cp ~/.config/colab-cli/token.json ~/.config/colab-cli/token-b.json

# 恢复账户 A
cp ~/.config/colab-cli/token-a.json ~/.config/colab-cli/token.json

# shell 函数快速切换
colab-switch() {
  case "$1" in
    a) cp ~/.config/colab-cli/token-a.json ~/.config/colab-cli/token.json ;;
    b) cp ~/.config/colab-cli/token-b.json ~/.config/colab-cli/token.json ;;
  esac
  echo "Switched to account $1"
}
```

**优点**：简单，不需要额外目录或 gcloud 配置
**缺点**：不能同时使用；token 过期需手动刷新；session 记录混在一起

### 方案四：--config + --auth 组合

仅分离 session 记录，token 仍共享。仅在以下场景有用：两个账户都用 ADC，通过 gcloud config 切换，但想保持 session 记录分开。

```bash
colab --auth adc --config ~/.config/colab-cli/sessions-a.json new -s training
```

**优点**：session 记录分开
**缺点**：token 仍冲突，不能解决 OAuth2 多账户问题

## 方案对比

| 方案 | 同时使用 | 隔离程度 | 设置成本 | 最佳场景 |
|------|---------|---------|---------|---------|
| 独立 HOME | ✅ | **完全** (token + session + log + settings + history) | 中 | 长期双账户 |
| 混合认证 OAuth2+ADC | ✅ | 部分 (token 隔离, session/log 共享) | 低 | 临时双账户 |
| Shell token 切换 | ❌ | 部分 (token 隔离, 其余共享) | 低 | 偶尔切换 |
| --config 分离 | ❌ | 低 (仅 session 隔离) | 低 | ADC + 清晰 session 管理 |

## 验证方法

```bash
# 查看当前认证的账户和 scope
colab whoami
# 输出:
#   Auth provider: oauth2
#   Email:         account-a@gmail.com
#   Expires in:    59m
#   Scopes: ...

# 方案一：分别验证
HOME=~/colab-accounts/account-a colab whoami
HOME=~/colab-accounts/account-b colab whoami

# 方案二：分别验证
colab --auth oauth2 whoami
colab --auth adc whoami
```

## 源码参考

所有结论基于 `google-colab-cli v0.5.9` 的完整源码阅读：

| 文件 | 关键内容 |
|------|---------|
| `auth.py:54` | `TOKEN_CONFIG_PATH` 硬编码 |
| `auth.py:190-210` | `get_credentials()` 入口，OAuth2 / ADC 分支 |
| `cli.py:52-78` | 全局 flags：`-c`, `--config`, `--auth` |
| `common.py:30-79` | `State` 单例，lazy 初始化 store / client / settings |
| `state.py:107-146` | `StateStore` 可接收自定义路径 |
| `state.py:83-105` | `SettingsStore` 默认路径硬编码 |
| `history.py:22-23` | `HistoryLogger` 默认路径硬编码 |
| `session.py:383-421` | `spawn_keep_alive()` 传递 `--auth` 和 `--config` 给守护进程 |
