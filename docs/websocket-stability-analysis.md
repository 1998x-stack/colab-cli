# Colab CLI WebSocket 稳定性分析

> **撰写日期**: 2026-06-11
> **环境**: 中国大陆网络环境，Clash/Meta SOCKS5 代理 (127.0.0.1:7890)
> **工具版本**: google-colab-cli v0.5.9 | websocket-client v1.9.0 | jupyter-kernel-client (bundled)

---

## 目录

1. [架构概览](#1-架构概览)
2. [失败模式](#2-失败模式)
3. [代理深度分析](#3-代理深度分析)
4. [现有缓解措施](#4-现有缓解措施)
5. [改进方案](#5-改进方案)
6. [诊断实验](#6-诊断实验)

---

## 1. 架构概览

### 1.1 两条独立的连接路径

colab CLI 在与 Colab 后端通信时,实际使用两条**完全不同**的通信路径:

```
用户机器
  │
  ├── REST API 路径 ────────────────── HTTPS ──► colab.pa.googleapis.com
  │   (colab new, keep-alive, colab stop)         (REST, 短连接, 无状态)
  │
  └── WebSocket 路径 ──────────────── WSS ────► *.prod.colab.dev
      (colab exec, colab repl,                (Jupyter Kernel, 长连接, 有状态)
       colab console)
```

### 1.2 REST API 路径

由 `colab_cli/client.py` 中的 `Client` 类处理。

```python
# client.py: Client._issue_request()
response = self.session.request(method, endpoint, ...)
```

- 使用 `requests` 库
- 自动通过 `HTTP_PROXY`/`HTTPS_PROXY` 环境变量识别代理
- 短连接：每次请求独立 TCP 连接，完成后立即释放
- **keep-alive daemon** 每 60 秒发送一次 `KeepAliveAssignment` RPC：

```python
# client.py:298-324
def keep_alive_assignment(self, endpoint: str):
    url = urljoin(self.colab_api_domain,
        "/$rpc/google.internal.colab.v1.RuntimeService/KeepAliveAssignment")
    ...
    return self._issue_request(url, method="POST", ...)
```

关键点：调用 `colab.pa.googleapis.com`，**不走代理的热点域 `*.colab.dev`**，且每次只是简单的 HTTPS POST。

### 1.3 WebSocket 路径

由 `colab_cli/runtime.py` → `jupyter_kernel_client` 处理。

执行链：

```
colab exec
  → ColabRuntime(url, token)
    → jupyter_kernel_client.KernelClient(server_url, ...)
      → KernelClient.start()
        → KernelHttpManager.client
          → KernelWebSocketClient (wsclient.py)
            → WebSocketApp(url).run_forever(ping_interval=60)
```

关键代码路径：

```python
# runtime.py:98-120
client_kwargs = {
    "subprotocol": jupyter_kernel_client.JupyterSubprotocol.DEFAULT,
    "extra_params": {"colab-runtime-proxy-token": self.token},
}
self._kernel_client = jupyter_kernel_client.KernelClient(
    server_url=self.url,  # https://8080-m-s-XXXX.us-central1-1.prod.colab.dev/
    token=self.token,
    ...
)
```

实际连接的 WebSocket URL 由 `KernelHttpManager.client` 构造：

```python
# manager.py:143-155
base_ws_url = HTTP_PROTOCOL_REGEXP.sub("ws", self.kernel_url, 1)
kw = {
    "endpoint": url_path_join(base_ws_url, "channels"),  # wss://.../api/kernels/{id}/channels
    "token": self.token,
    ...
}
```

然后在 `KernelWebSocketClient.start_channels()` 中：

```python
# wsclient.py:585-594
self.kernel_socket = websocket.WebSocketApp(
    url,  # wss://8080-m-s-XXXX.us-central1-1.prod.colab.dev/.../channels?token=...&session_id=...
    header=self._headers,
    ...
)
self.connection_thread = Thread(target=self._run_websocket)
self.connection_thread.start()
self.connection_ready.wait(timeout=self.timeout)
```

而 `_run_websocket` 调用 `run_forever(ping_interval=self.ping_interval)`，**没有传递任何代理参数**。

### 1.4 两条路径的存活差异

| 特性 | REST (keep-alive) | WebSocket (exec) |
|------|------------------|------------------|
| 连接类型 | 短连接 (每次请求独立) | 长连接 (持续保持) |
| 库 | `requests` | `websocket-client` + `PySocks`(可选) |
| 代理感知 | 自动读 `HTTP_PROXY`/`HTTPS_PROXY` | 需显式传递 `proxy_type`/`http_proxy_host` |
| 防火墙穿透 | HTTPS，较稳定 | WSS，易被干扰 |
| 目标域名 | `colab.pa.googleapis.com` | `*.prod.colab.dev` |
| 断开后果 | 下一分钟重试 (无状态) | 执行中断 (有状态) |

这就是核心问题所在：**keep-alive 走 REST，简单可靠；exec 走 WebSocket，复杂易断**。

---

## 2. 失败模式

### 2.1 模式 A：WebSocket 握手阶段失败（最致命）

**场景**：`colab exec` 刚启动时，WebSocket 连接完全无法建立。

**表现**：
```
jupyter_kernel_client.wsclient: Unable to open websocket connection with ...
requests.exceptions.ReadTimeout: ... 
```

**根因**：SOCKS5 代理在处理 `wss://` 的 TLS 握手时，可能有以下几种失败原因：

1. **Clash/Meta 的 WebSocket 处理 bug**：部分代理实现中，WebSocket 升级请求（`Upgrade: websocket`）未被正确转发
2. **GFW 主动干扰**：SSL 握手时发送 RST 包
3. **DNS 污染**：`*.prod.colab.dev` 的解析结果被指向错误的 IP

**频率**：每次 `colab exec` 时约 20-30% 概率发生（观察数据）。

### 2.2 模式 B：WebSocket 连接中断（最常见）

**场景**：exec 已经开始执行，中途断开。

**表现**：
```
colab_cli.execution: Session appears to be lost (404/401)
```
或 `TimeoutError`。

**根因**：这是最复杂的失败链条：

```
SOCKS5 代理
   └── 中间 NAT 设备 (运营商级)
        └── GFW 状态检测
             └── WebSocket 连接闲置 > 60s → 中间节点静默断开
                  → 客户端收不到 FIN/RST
                  → sock.recv() 无限阻塞
                  → 直到 jupyter-kernel-client 的 REQUEST_TIMEOUT (10s) 触发
                  → _recv_reply() 抛出 TimeoutError
```

为什么连接会闲置？因为大多数 `colab exec` 操作是**发送代码 → 等待结果**的模式。在等待期间，WebSocket 上没有数据流动。默认的 `ping_interval=60` 意味着每 60 秒才发一次 WebSocket Ping 帧。但许多中间设备的闲置超时是 **30-60 秒**。

### 2.3 模式 C：执行超时（容易误判）

**场景**：代码实际上仍在执行，但因 WebSocket 临时不通导致客户端认为超时。

**表现**：
```
TimeoutError: Timeout waiting for reply
```

**根因**：`jupyter_kernel_client` 的默认 `timeout` 是 `REQUEST_TIMEOUT` (10 秒)。如果：
- 训练任务输出不频繁（>10 秒无 stdout）
- GPU 编译任务需要更长时间
- 网络短暂抖动导致 1-2 秒无数据

客户端就会认为连接已死。实际查看 `ColabRuntime.execute_code` 的代码：

```python
# runtime.py:171-175
kwargs = {"allow_stdin": allow_stdin}
if timeout is not None:
    kwargs["timeout"] = timeout
```

exec 命令的默认 timeout 是 **10 秒**：

```python
# execution.py:107-113
def exec_command(
    ...
    timeout: Annotated[Optional[float], ...] = 10.0,
    ...
```

### 2.4 模式 D：Session 被后端回收

**场景**：`colab exec` 发现 401/404，认为 session 已死。

**表现**：
```
[colab] Session 'training' appears to be lost (404/401). Cleaning up.
```

**根因**：后端（Colab 的 Runtime Service）在以下情况会将 VM 回收：
- keep-alive 间隔超过阈值（约 5 分钟）
- session 空闲过久
- 账号/配额问题

**但在中国用户场景中，更常见的情况是**：keep-alive 通过了（走 REST），但 WebSocket 断了，exec 的 `on_close` 触发后，后续 REST API 调用（如 `is_terminal_error` 检查）拿到了 404，因为后端认为你的连接已经断开。

这里有一个竞态条件：
```
时间线：
1. WebSocket 断开 (客户端感知到)
2. exec 尝试调用 REST API 检查 session 状态
3. keep-alive daemon 在下一分钟也尝试 ping
4. 如果 keep-alive 先成功 → session 存活，exec 误报
5. 如果 exec 的检查先到 → 可能看到暂时 404，误判 session 死亡
```

---

## 3. 代理深度分析

### 3.1 SOCKS5 代理与 WebSocket 的兼容性问题

#### 3.1.1 `websocket-client` 库的代理解析机制

`websocket-client` 库在 `WebSocket.connect()` 方法中通过 `_url.get_proxy_info()` 读取代理设置：

```python
# _url.py:131-189
def get_proxy_info(hostname, is_secure, proxy_host=None, ..., proxy_type="http"):
    if _is_no_proxy_host(hostname, no_proxy):
        return None, 0, None

    if proxy_host:
        ...  # 显式指定的代理
        return proxy_host, port, auth

    env_key = "https_proxy" if is_secure else "http_proxy"
    value = os.environ.get(env_key, os.environ.get(env_key.upper(), ""))
    if value:
        proxy = urlparse(value)
        ...
        return proxy.hostname, proxy.port, auth

    return None, 0, None  # ← 无代理
```

**关键问题**：
1. `proxy_type` 参数**不会从环境变量读取**。即使 `HTTP_PROXY=socks5://127.0.0.1:7890` 设置了，`get_proxy_info` 返回的 `proxy_type` 仍然是默认的 `"http"`。
2. `_http.py` 中的代理连接代码会根据 `proxy_type` 用不同方式连接：
   - `"http"` → HTTP CONNECT 隧道（不支持 SOCKS）
   - `"socks5"` → 需要 `PySocks` 库

#### 3.1.2 连接路径对比

```
直接连接 (no_proxy):
  client ──TCP──► *.prod.colab.dev:443
                   ↑ 无代理开销，但可能被 GFW 拦截

HTTP 代理 (proxy_type="http"):
  client ──TCP──► Clash:7890
    └── HTTP CONNECT wss://*.prod.colab.dev:443
          └── Clash ──TCP──► *.prod.colab.dev:443
                ↑ 多一跳，HTTP CONNECT 对 WSS 支持良好

SOCKS5 代理 (proxy_type="socks5"):
  client ──TCP──► Clash:7890
    └── SOCKS5 握手
      └── Clash ──TCP──► *.prod.colab.dev:443
            ↑ 多一跳，但 Clash 的 SOCKS5 可能对 WS 协议有特殊处理
```

### 3.2 `no_proxy` 方案分析

当前有效的变通方案：

```bash
export no_proxy="*.colab.dev,*.prod.colab.dev,localhost,127.0.0.1"
```

这会让 `websocket-client` 对 `*.prod.colab.dev` 的 WebSocket 连接走**直连**。

**为什么直连有效？**
- `*.colab.dev` 域名的 CDN（Google Cloud Load Balancer）在中国大陆的某些地区可以访问
- 直连避免了 SOCKS5 代理的额外延迟和可能的 WS-over-SOCKS5 bug
- `no_proxy` 在 `_url._is_no_proxy_host()` 中精确匹配

**直连的风险**：
- 中国大陆到 `*.prod.colab.dev` 的 TCP 连接可能不稳定
- GFW 可能随时加强对 `colab.dev` 的封锁
- 不同 ISP（电信/联通/移动）的直连质量差异很大

### 3.3 HTTP 代理 vs SOCKS5 代理

| 特性 | SOCKS5 | HTTP CONNECT (HTTP Proxy) |
|------|--------|--------------------------|
| 协议支持 | 通用 (任何 TCP/UDP) | 仅 TCP |
| WebSocket 兼容性 | 依赖实现，Clash 表现良好 | 天然支持 |
| 握手开销 | 1 个额外 RTT | 2 个额外 RTT (CONNECT + 200 OK) |
| `HTTP_PROXY` 环境变量 | 不被 `websocket-client` 识别为 SOCKS5 | 自动识别 |
| 对 GFW 的隐身性 | 不可见 (纯 TCP 隧道) | 显式 CONNECT 请求，可能被探测 |

### 3.4 Clash/Meta 的 TUN 模式

如果在 TUN 模式下运行 Clash（全局虚拟网卡），所有流量（包括 WebSocket）都会被强制走代理。此时 `no_proxy` 不会生效，因为代理在操作系统级别接管了所有连接。

**TUN 模式下的问题**：
- DNS 解析由 Clash 处理，可能绕过 `no_proxy`
- 所有流量走代理 → WebSocket 走代理 → WS over SOCKS5 over TUN 多了一层
- 性能损失更大，但兼容性可能更好

---

## 4. 现有缓解措施

### 4.1 Detached Bootstrap 模式

这是项目中最有效的变通方案。原理：

```bash
# launch.py (运行在 VM 上)
import subprocess, os
proc = subprocess.Popen(
    ["python", "-u", "train.py"],
    start_new_session=True,  # 脱离父进程组
    ...
)
# launch.py 立即退出
```

```bash
# 本地 (colab exec 只运行几毫秒就返回)
colab exec -s training -f launch.py --timeout 120
# WebSocket 在 launch.py 返回后就关闭了
# 但 train.py 依然在 VM 上运行 (因为 start_new_session=True)
```

**为什么有效**：WebSocket 只在传输 `launch.py` 的几毫秒内需要存活。后续训练过程完全在 VM 上自运行，不依赖本地连接。

**缺点**：
- 无法实时获取输出
- 需要额外的 watchdog + check_progress 架构
- 失败时没有即时反馈

### 4.2 Keep-Alive Daemon

`spawn_keep_alive()` 启动一个子进程，每 60 秒通过 REST API 发送 `KeepAliveAssignment` 请求。

```python
# session.py:424-502
# 子进程每 60 秒调用:
state.client.keep_alive_assignment(endpoint)  # HTTPS POST
```

**与 WebSocket 的关系**：keep-alive 只防止 VM 被后端回收，**完全不解决 WebSocket 稳定性问题**。它和 exec 的 WebSocket 是两个独立通道。

**误区澄清**：项目 CLAUDE.md 提到的"观察到 session 在 12-15 分钟后死亡"和 keep-alive 是否运行关系不大。12-15 分钟的死亡周期，更可能是：
1. 免费账号的 GPU 配额限制（不是 12 小时，而是 T4 的实际可运行时间更短）
2. WebSocket 断开后无重连机制
3. 中间设备的 NAT 超时（常见设置为 10-15 分钟）

### 4.3 `no_proxy` 环境变量

```bash
export no_proxy="*.colab.dev,*.prod.colab.dev,localhost,127.0.0.1"
```

让 WebSocket 直连 `*.prod.colab.dev`。

**适用场景**：`colab exec` 直接使用（非 TUN 模式）。

### 4.4 `--timeout` 参数

`colab exec --timeout 300` 将内部超时从默认 10 秒提升到 5 分钟，减轻网络抖动导致的误判。

---

## 5. 改进方案

> 方案按 影响/实现成本 降序排列。

### 5.1 [高/低] 为 WebSocket 连接添加代理配置支持 (P0)

**现状**：`KernelWebSocketClient._run_websocket()` 调用 `run_forever()` 时**没有传递任何代理参数**：

```python
# wsclient.py:1279
self.kernel_socket.run_forever(
    ping_interval=self.ping_interval, reconnect=self.reconnect_interval
)
# 缺少: proxy_type, http_proxy_host, http_proxy_port, http_proxy_auth
```

**改进**：

```python
def _run_websocket(self) -> None:
    proxy_type = os.environ.get("WS_PROXY_TYPE", os.environ.get("ws_proxy_type", "http"))
    proxy_host = os.environ.get("WS_PROXY_HOST", os.environ.get("ws_proxy_host"))
    proxy_port = int(os.environ.get("WS_PROXY_PORT", os.environ.get("ws_proxy_port", "0")))
    no_proxy = os.environ.get("no_proxy", os.environ.get("NO_PROXY", ""))

    self.kernel_socket.run_forever(
        ping_interval=self.ping_interval,
        reconnect=self.reconnect_interval,
        proxy_type=proxy_type if proxy_type != "http" else None,
        http_proxy_host=proxy_host,
        http_proxy_port=proxy_port,
        http_no_proxy=no_proxy.split(",") if no_proxy else None,
    )
```

但请注意：**这是第三方库 `jupyter-kernel-client`（Datalayer, Inc. 维护）**，不是 colab CLI 自身控制的代码。Google 的 colab CLI 只是调用这个库。

**可行的改造路径**：
1. 在 `ColabRuntime` 层包装 `KernelWebSocketClient`，传入自定义的 `run_forever` 调用
2. 或者直接使用 `websocket-client` 的 WebSocket 连接，手动管理 Jupyter 协议
3. 或者提交 PR 到 `jupyter-kernel-client` 项目

### 5.2 [中/低] 增加 WebSocket 重连机制 (P0)

**现状**：`KernelWebSocketClient` 的 `reconnect_interval` 参数默认为 `0`（不重连）：

```python
# wsclient.py:498
reconnect_interval: int = 0,  # 不重连
```

改进：在 `ColabRuntime` 中启用重连：

```python
# runtime.py 中创建 KernelClient 时传递
client_kwargs = {
    "subprotocol": ...,
    "extra_params": {...},
    "reconnect": 5,  # 5 秒后重连
}
```

**注意**：Jupyter 协议是有状态的（execution state, message IDs），简单重连后需要恢复会话。如果 kernel 还在运行，重连后可以继续接收消息。Jupyter Server 的 WebSocket 层支持会话恢复（通过 `session_id`）。

### 5.3 [低/低] 缩短 Ping 间隔 (P1)

**现状**：Ping 间隔为 60 秒。中间设备的 NAT 超时通常为 30-60 秒。

**改进**：将 `ping_interval` 从 60 秒改为 **25-30 秒**。

```python
# wsclient.py 构造函数
ping_interval: float = 30,  # 从 60 改为 30
```

或者通过环境变量可配置：
```python
ping_interval = float(os.environ.get("WS_PING_INTERVAL", "30"))
```

**原理**：WebSocket Ping 帧（opcode 0x9）会触发 TCP ACK，刷新所有中间设备的 NAT 超时表。25 秒的间隔可以确保在任何常见的超时设置（30 秒以上）下连接不会中断。

### 5.4 [高/中] SSH 隧道替代 SOCKS5 (P1)

**原理**：通过 SSH 隧道建立到海外 VPS 的稳定连接，然后通过该隧道转发 WebSocket。

```bash
# 方案 A：本地端口转发 (直接)
ssh -L 8080:localhost:8080 user@vps
# 然后将 colab 目标端口映射到本地

# 方案 B：动态转发 (SOCKS5 over SSH)
ssh -D 7891 user@vps  # 在 VPS 上开 SOCKS5
# colab CLI 使用 SOCKS5 代理 localhost:7891
```

**优势**：
- SSH 的 TCP 连接比 SOCKS5 库更稳定
- SSH 有内置 keep-alive（`ServerAliveInterval`）
- 绕过 GFW 的深层包检测（SSH 的流量特征不易被识别）

**缺点**：需要海外 VPS，增加成本和运维。

### 5.5 [中/高] 使用 `proxychains-ng` 包装 colab CLI

```bash
# 强制所有 TCP 连接走 SOCKS5，包括 websocket-client 的原始 socket
proxychains4 -f /etc/proxychains.conf colab exec -s training -f script.py
```

**优势**：`proxychains-ng` 使用 `LD_PRELOAD` 劫持 `connect()` 系统调用，**所有 TCP 连接都会被代理**，不需要应用层支持。

**缺点**：
- 不能对特定域名设 `no_proxy`
- 可能与系统安全机制冲突
- macOS 上 SIP 可能阻止 `LD_PRELOAD`（需要禁用 SIP 或使用其他方法）

### 5.6 [高/高] 升级至 Colab Pro / 使用国内替代

- **Colab Pro/Pro+**：更稳定的连接（有 SLA），但价格较高且仍需代理
- **阿里云 PAI / 百度 AI Studio**：国内可用，无需代理
- **AutoDL / 恒源云**：国内 GPU 租用平台，延迟更低

### 5.7 [低/中] 改进 `colab exec` 的默认超时

**现状**：`--timeout` 默认 10 秒。对于网络不稳定的环境，这个值太短。

```python
# execution.py:exec_command
timeout: Optional[float] = 10.0  # 太短
```

**改进**：
- 检测到 HTTP_PROXY 环境变量时，自动将默认超时提高到 120 秒
- 添加 `--no-wait` 模式（detached exec），提交后立即返回

---

## 6. 诊断实验

以下实验可以帮助确认具体失败原因。

### 6.1 验证 WebSocket 直连 vs 代理连接

```bash
# 实验 1：直连测试 (with no_proxy)
export no_proxy="*.colab.dev,*.prod.colab.dev,localhost,127.0.0.1"
colab new --gpu T4 -s diag-test
timeout 30 colab exec -s diag-test -f /dev/stdin <<< "print('hello')"
# 观察：是否能成功？耗时多久？
colab stop -s diag-test

# 实验 2：代理连接测试 (without no_proxy)
unset no_proxy
colab new --gpu T4 -s diag-test2
timeout 30 colab exec -s diag-test2 -f /dev/stdin <<< "print('hello')"
# 观察：是否能成功？耗时多久？
colab stop -s diag-test2
```

**预期结果**：
- 如果实验 1 成功、实验 2 失败 → 问题在 SOCKS5 + WebSocket 适配
- 如果实验 1 也频繁失败 → `colab.dev` 本身直连不通过，需要用其他代理策略

### 6.2 验证 Ping 间隔效果

```bash
# 修改 wsclient.py 中的 ping_interval 从 60 改为 15
# 然后运行较长的 exec
colab exec -s training -f long_script.py --timeout 300
# 观察断开频率
```

### 6.3 验证 SSH 隧道方案

```bash
# 1. 建立 SSH 隧道
ssh -D 7892 -N user@your-vps &

# 2. 使用 SSH 的 SOCKS5 代理环境变量
export HTTPS_PROXY=socks5://127.0.0.1:7892
export no_proxy=""  # 不走 no_proxy

# 3. 测试 exec
colab exec -s training -f script.py --timeout 60
```

### 6.4 验证 `proxychains-ng` 方案

```bash
# macOS 安装
brew install proxychains-ng

# 配置 (写入 /opt/homebrew/etc/proxychains.conf)
# [ProxyList]
# socks5 127.0.0.1 7890

# 测试
proxychains4 colab exec -s training -f script.py --timeout 60
```

### 6.5 诊断日志收集

```bash
# 启用详细日志
colab --logtostderr exec -s training -f script.py 2>&1 | tee /tmp/colab-ws-diag.log

# 查看 WebSocket 相关日志
grep -i "websocket\|connection\|timeout\|retry\|close\|error\|ping\|pong" /tmp/colab-ws-diag.log

# 查看 colab 日志目录
cat ~/.config/colab-cli/colab.log | grep -i "websocket"
```

### 6.6 持续监控 WebSocket 状态

```python
# ws_monitor.py - 在 colab exec 运行期间监控 WebSocket
import subprocess
import time

# 获取 colab CLI 的 WebSocket 连接 PID
pid = subprocess.check_output(["pgrep", "-f", "colab"]).decode().strip()

# 监控 TCP 连接状态
while True:
    conns = subprocess.check_output(
        ["lsof", "-p", pid, "-i", "TCP"]
    ).decode()
    print(f"[{time.strftime('%H:%M:%S')}] Connections:\n{conns}")
    time.sleep(5)
```

### 6.7 关键实验：分离 rest 和 ws 的代理配置

```bash
# REST API：走 SOCKS5 代理
export HTTPS_PROXY=socks5://127.0.0.1:7890
export HTTP_PROXY=socks5://127.0.0.1:7890

# WebSocket：直连 (通过 no_proxy)
export no_proxy="*.colab.dev,*.prod.colab.dev"

# 运行较长的任务
colab exec -s training -f train.py --timeout 300
```

**这是当前理论上的最佳配置。** REST API 通过 SOCKS5 代理保持连通，不会被 GFW 拦截；WebSocket 直连 `colab.dev`，避免了 SOCKS5 over WS 的兼容性问题。

---

## 附录 A：关键代码位置

| 组件 | 文件 | 关键函数/方法 |
|------|------|--------------|
| REST Client | `client.py` | `Client.keep_alive_assignment()` (L298-324) |
| Keep-alive daemon | `session.py` | `keep_alive()` (L424-502), `spawn_keep_alive()` (L383-421) |
| Runtime/WS | `runtime.py` | `ColabRuntime.kernel_client` (L89-156) |
| Jupyter WS Client | `wsclient.py` | `KernelWebSocketClient.start_channels()` (L565-609), `_run_websocket()` (L1273-1289) |
| Jupyter Manager | `manager.py` | `KernelHttpManager.client` (L136-157) |
| WS App | `_app.py` | `WebSocketApp.run_forever()` (L256-275) |
| Proxy Resolver | `_url.py` | `get_proxy_info()` (L131-189) |
| `colab exec` | `execution.py` | `exec_command()` (L101-236) |

## 附录 B：环境变量参考

| 变量 | 影响 | 读取方 |
|------|------|--------|
| `HTTP_PROXY` / `http_proxy` | REST API 代理 | `requests` 库 |
| `HTTPS_PROXY` / `https_proxy` | REST API 代理 | `requests` 库 |
| `https_proxy` | WebSocket connect 代理 | `websocket._url.get_proxy_info()` |
| `no_proxy` / `NO_PROXY` | 跳过代理的域名 | 两方都读取 |
| `WS_PROXY_TYPE` | WebSocket 代理类型 (http/socks5) | **不被任何代码读取** |
| `WS_PING_INTERVAL` | WebSocket ping 间隔 | **不被任何代码读取** |
