# Colab 会话健康监控与自动恢复策略分析

## 目录

1. [会话生命周期](#1-会话生命周期)
2. [故障模式与检测方法](#2-故障模式与检测方法)
3. [现有监控体系分析](#3-现有监控体系分析)
4. [自动恢复架构设计](#4-自动恢复架构设计)
5. [多账号轮转策略](#5-多账号轮转策略)
6. [实施优先级](#6-实施优先级)

---

## 1. 会话生命周期

### 1.1 状态机

```
         ┌──────────────────────────────────────────┐
         │                                          │
         v                                          │
   ┌──────────┐   colab new --gpu T4   ┌─────────────────────┐
   │ 不存在    │ ──────────────────────>│ 预检 (pre-flight)    │
   │ (NoSession)│                      │ (keep_alive RPC测试) │
   └──────────┘                        └──────────┬──────────┘
                                                  │
                                ┌─────────────────┼─────────────────┐
                                │ 403 SCOPE_ERR   │ 成功             │ 其他错误
                                v                 v                  v
                          ┌───────────┐    ┌──────────────┐    ┌──────────┐
                          │ 拒绝创建   │    │ 已分配         │    │ 继续创建  │
                          │ + 清理资源  │    │ (Assigned)    │    │ (daemon  │
                          └───────────┘    └──────┬───────┘    │  重试)   │
                                                  │            └──────────┘
                                                  │ spawn_keep_alive()
                                                  v
                                          ┌──────────────┐
                                          │ 运行中        │
                                          │ (Running)    │
                                          │ keep_alive   │
                                          │ daemon: 60s  │
                                          └──────┬───────┘
                                                  │
                    ┌─────────────────────────────┼─────────────────────────────┐
                    │                             │                             │
                    v                             v                             v
          ┌──────────────────┐     ┌──────────────────────┐    ┌────────────────────┐
          │ 正常结束          │     │ 故障断开              │    │ 本地清理            │
          │ colab stop       │     │ (见 2.1-2.3)         │    │ state.prune_session │
          │ unassign RPC     │     │                      │    │ (sync_sessions发现  │
          │ + 杀 daemon      │     │ daemon: 2次4xx退出    │    │ endpoint不存在于    │
          └──────────────────┘     │ 或 24h 超时           │    │ server列表)         │
                                   └──────┬───────────────┘    └────────────────────┘
                                          v
                                  ┌──────────────────┐
                                  │ 变成孤儿 (Orphan) │
                                  │ server仍有记录     │
                                  │ 本地已同步清理      │
                                  │ 或 daemon已退出     │
                                  └──────────────────┘
```

### 1.2 核心数据结构

`sessions.json` (~/.config/colab-cli/sessions.json):

```json
{
  "session-name": {
    "name": "transformer-baseline",
    "token": "eyJhbG...",              // runtime proxy token
    "url": "https://8080-...prod.colab.dev",
    "endpoint": "gpu-t4-s-...",         // 全局唯一 endpoint ID
    "variant": "GPU",
    "accelerator": "T4",
    "kernel_id": "4420f2ff-...",        // Jupyter kernel ID
    "session_id": "f2bbb952-...",       // Jupyter session ID
    "last_execution": ["stdin", null, "2026-06-11 15:10:22"],
    "running": null,                    // 当前正在执行的操作描述
    "keep_alive_pid": 25356             // 后台 daemon 进程 PID
  }
}
```

### 1.3 Keep-Alive Daemon 内部逻辑

`session.py` 中的 `keep_alive()` 函数：

```
循环 (最多 24 小时):
  1. sleep(60s)
  2. 从 sessions.json 检查 session 是否仍存在
  3. 如果 endpoint 已变更 → exit('endpoint_mismatch')
  4. 调用 keep_alive_assignment(endpoint) RPC
  5. 如果成功 → consecutive_4xx = 0
  6. 如果 4xx 错误 → consecutive_4xx++
     - consecutive_4xx >= 2 → exit('consecutive_4xx_errors')
  7. 如果网络错误 (非 4xx) → 继续重试，不计数
```

**退出原因：**
- `time_limit_reached` — 24 小时上限
- `session_not_found` — 本地 sessions.json 中已删除
- `endpoint_mismatch` — endpoint 已变更
- `consecutive_4xx_errors` — 连续 2 次 4xx 错误（通常表示会话已死）

### 1.4 会话同步机制

`state.sync_sessions()` 流程：

```
1. 读取本地 sessions.json
2. 调用 list_assignments() 获取服务端活跃 assignments
3. 如果本地 session 的 endpoint 不在服务端列表中 → 视为已清理
4. 调用 prune_session(name):
   - 杀死对应的 keep_alive_pid
   - 从 sessions.json 删除
   - 记录 history 事件
```

---

## 2. 故障模式与检测方法

### 2.1 WebSocket 断开（通过代理）

**现象：**
- `colab exec` 时报 `ReadTimeout` 或 `ConnectTimeout`
- kernel 连接无法建立（`jupyter_kernel_client` 内部 WebSocket 断开）
- daemon 日志中出现非 4xx 网络错误

**根本原因：**
- 位于中国，Clash 等代理对 WebSocket 连接不稳定
- 某些区域 (asia-southeast1, europe-west4) 的 WebSocket 容易因网络策略中断
- 代理超时时间短于 Colab 的 ping_interval (30s)

**检测方法：**

```python
# 方法 1: 尝试连接 kernel，捕捉特定异常
from colab_cli.utils import is_terminal_error
try:
    runtime.execute_code("import os")
except Exception as e:
    if is_terminal_error(e):
        # 401/404 — 会话已死
        pass
    elif "Timeout" in type(e).__name__:
        # 网络超时 — 可能是 WS 断开，也可能是暂时网络抖动
        pass
    elif "ConnectionRefused" in str(e):
        # 连接被拒绝
        pass

# 方法 2: HTTP 层面探活
import requests
url = f"{session.url}/api/kernels/{session.kernel_id}"
headers = {"X-Colab-Runtime-Proxy-Token": session.token}
try:
    r = requests.get(url, timeout=5,
        headers=headers,
        params={"authuser": "0", "colab-runtime-proxy-token": session.token})
    if r.status_code == 200:
        # 内核存活
        pass
    elif r.status_code in (401, 404):
        # 会话已死
        pass
except requests.Timeout:
    # 网络层问题
    pass
```

**关键观察：** 网络错误和真正的会话死亡难以区分。daemon 只通过 `consecutive_4xx_errors` 来判断——非 4xx 错误（网络超时、DNS 失败等）会被忽略并重试。这意味着网络抖动不会导致 daemon 退出，但长时间的 WebSocket 断开也**不会**被 daemon 检测到（因为它不返回 HTTP 错误代码，而是连接超时）。

### 2.2 GPU 配额耗尽

**现象：**
- `colab new --gpu T4` 返回 400 错误
- 分配了但 GPU 不可用（`nvidia-smi` 失败）
- VM 成功创建但 GPU 为 NONE（降级为 CPU）

**根本原因：**
- 免费账号每日 GPU 配额约 10-12 小时，动态调整
- 频繁使用后冷却期可达 12-24 小时甚至数周
- GPU 限制在**分配阶段**生效，运行期间不会丢失 GPU

**检测方法：**

```python
# 方法 1: 解析 colab new 的错误
# TooManyAssignmentsError — 账号已有一个活跃分配
# 400 error with accelerator — GPU 配额不足

# 方法 2: VM 上验证 GPU 是否真的可用
def check_gpu_on_vm(session_name):
    """通过 colab exec 检查 GPU 状态"""
    code = """
import subprocess, torch
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
    print(f"nvidia-smi: {'OK' if result.returncode == 0 else 'FAILED'}")
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
except Exception as e:
    print(f"Error: {e}")
"""
    subprocess.run(["colab", "exec", "-s", session_name, "-f", "/dev/stdin"],
                   input=code, text=True, timeout=30)

# 方法 3: 记录账号级别的 GPU 使用历史
import json, os, time
from datetime import datetime

USAGE_LOG = os.path.expanduser("~/.config/colab-cli/gpu_usage.json")

def log_gpu_usage(account_alias: str, action: str, success: bool):
    """记录 GPU 使用事件"""
    log = []
    if os.path.exists(USAGE_LOG):
        with open(USAGE_LOG) as f:
            log = json.load(f)
    log.append({
        "account": account_alias,
        "action": action,       # "assign", "unassign", "check"
        "success": success,
        "timestamp": datetime.utcnow().isoformat(),
    })
    # 只保留最近 100 条
    log = log[-100:]
    with open(USAGE_LOG, "w") as f:
        json.dump(log, f, indent=2)

def get_account_cooldown(account_alias: str) -> float:
    """返回账号距离上次 GPU 使用的冷却时间（小时）"""
    if not os.path.exists(USAGE_LOG):
        return 0.0
    with open(USAGE_LOG) as f:
        log = json.load(f)
    events = [e for e in log if e["account"] == account_alias and e["action"] == "assign"]
    if not events:
        return 0.0
    last = events[-1]
    last_time = datetime.fromisoformat(last["timestamp"])
    elapsed = (datetime.utcnow() - last_time).total_seconds() / 3600
    return max(0.0, 12.0 - elapsed)  # 假设最少冷却 12 小时
```

### 2.3 会话被静默清理（Session Pruned）

**现象：**
- `colab sessions` 突然看不到某个会话
- 之前正常运行的会话消失
- daemon 日志显示 `session_not_found`

**根本原因：**
- `state.sync_sessions()` 在发现 session endpoint 不在服务端列表时自动清理
- 服务端可能由于空闲超时、内部错误、或后端维护清理了会话
- 调用 `colab sessions` 或 `colab status` 时触发了同步

**检测：**

```python
# daemon 会在退出时记录 reason=session_not_found
# 查看 history 日志:
cat ~/.config/colab-cli/history/<session_name>.jsonl | grep keep_alive_stopped

# 输出示例:
{"timestamp": "...", "event_type": "keep_alive_stopped",
 "reason": "session_not_found", "iterations": 42, ...}
```

**与其它故障的区别：**

| 故障模式 | daemon 退出原因 | colab sessions | colab exec |
|---------|----------------|---------------|-----------|
| WebSocket 断开 | 不会退出（非 4xx 重试） | 可能仍显示 | 超时/失败 |
| GPU 配额耗尽 | 不适用（无法创建） | 无此会话 | 不适用 |
| 会话被清理 | `session_not_found` | 已清除 | "Session not found" |
| 服务端 4xx | `consecutive_4xx_errors` | 可能仍显示* | 失败 |

> *注意：sync_sessions() 只在调用 sessions/status 命令时触发，daemon 不主动同步

### 2.4 检测方法总结

| 层级 | 检测手段 | 延迟 | 可靠性 |
|------|---------|------|-------|
| **daemon RPC** | keep_alive_assignment 成功/失败 | ~60s | 高（HTTP 层面） |
| **kernel API** | GET /api/kernels/{id} | 即时 | 中（网络依赖） |
| **exec 执行** | execute_code 超时/异常 | 执行时 | 高（端到端） |
| **sync 同步** | list_assignments 对比 | 调用时 | 高（服务端状态） |
| **VM 心跳** | watchdog/heartbeat.json | ~30s-2min | 中（需文件系统可达） |
| **进程检查** | pgrep -f train.py | 即时 | 中（仅 VM 内部） |

---

## 3. 现有监控体系分析

### 3.1 项目现有架构

```
┌─────────────────────────────────────────────┐
│           本地机器 (Local)                    │
│                                              │
│  colab exec -f check_progress.py             │
│       │                                      │
│       │ (通过 cron 每 5-7 分钟)              │
│       v                                      │
│  读取 VM 文件 (/content/heartbeat.json)       │
│  或 /content/metrics.jsonl                    │
│  或 /content/train.log                        │
│                                              │
│  CronCreate 定时任务                          │
└─────────────────────┬───────────────────────┘
                      │ colab exec (HTTP/WS)
                      v
┌─────────────────────────────────────────────┐
│           Colab VM                           │
│                                              │
│  watchdog.py (每 30s 写心跳)                 │
│  train.py (每 epoch 写指标)                  │
│  launch.py (启动时创建 /content/watchdog_stop)│
└─────────────────────────────────────────────┘
```

### 3.2 各项目文件对比

**transformer_iwslt/check_progress.py**
- 读取 `/content/metrics.jsonl`（train.py 每 epoch 写入）
- 检查 train.py 进程是否存活（`pgrep -f train.py`）
- 报告最新 epoch、loss、BLEU、LR
- 告警：进程死但 epoch<20（CRITICAL）、loss>8（WARNING）

**alexnet_imagenette/check_progress.py**
- 读取 `/content/heartbeat.json`（watchdog.py 每 30s 更新）
- 检查 train.py 进程是否存活
- 检测心跳过期（>120s 判定为 stale）
- 告警：心跳 stale + 无进程（CRITICAL）、>55min 触发紧急下载

**alexnet_imagenette/watchdog.py**
- VM 端守护进程，每 30s 更新 heartbeat.json 的 `watchdog_seen` 时间戳
- 检查 `/content/watchdog_stop` 文件是否存在（train.py 完成时创建）
- 训练结束后写入最终心跳

**vllm-compare/check_progress.py**
- 最简单的实现：仅检查 `pgrep -f compare.py` 和读取 log 末尾 20 行

**nanogpt/check_progress.py**
- 检查进程、日志末尾、checkpoint 文件、生成的图表

### 3.3 现有监控的缺陷

1. **依赖 `colab exec` 可达性** — 如果 WebSocket 断开，check_progress.py 根本无法执行
2. **被动模式** — 监控是 cron 轮询，没有主动推送告警
3. **缺少会话级健康检测** — 只检查 VM 内进程，不检查 colab 会话是否存活
4. **无自动恢复** — 检测到失败后只会打印告警，不会尝试恢复
5. **心跳延迟** — watchdog 间隔 30s + cron 间隔 5min = 最坏情况 5.5min 才能发现会话死亡
6. **无账号轮转** — 多账号需要手动切换 HOME 环境变量

---

## 4. 自动恢复架构设计

### 4.1 架构概述

```
┌──────────────────────────────────────────────────────┐
│               Session Health Manager (SHM)             │
│               python session_health_manager.py         │
│                                                        │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Probe Layer  │  │ Detect Layer │  │ Recover Layer│  │
│  │              │  │              │  │              │  │
│  │ • RPC探活    │→│ • 故障分类   │→│ • 重分配     │  │
│  │ • Kernel API │  │ • 误报过滤   │  │ • 账号轮转   │  │
│  │ • HTTP探活   │  │ • 严重性判定  │  │ • 断点续训   │  │
│  └─────────────┘  └──────────────┘  └──────────────┘  │
│                                                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │            State & History Store                  │  │
│  │  ~/.config/colab-cli/shm_state.json               │  │
│  └──────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
         │
         │ 通过 cron 每 2 分钟运行
         v
┌──────────────────────────────────────────────────────┐
│                  Notification                         │
│  • 终端输出 (当前实现)                                │
│  • macOS 通知 (osascript)                              │
│  • 微信/Telegram 推送 (可选)                           │
└──────────────────────────────────────────────────────┘
```

### 4.2 Probe Layer — 多层级探活

```python
"""session_probe.py — 会话健康探测"""

import requests
import subprocess
import json
import time
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

@dataclass
class ProbeResult:
    alive: bool              # 会话是否存活
    probe_type: str          # "rpc", "kernel_api", "exec", "sync"
    latency_ms: float        # 探测延迟
    error: Optional[str] = None
    detail: Dict[str, Any] = field(default_factory=dict)

class SessionProbe:
    """多层级会话探活器"""

    def __init__(self, state_store, history_store):
        self.store = state_store
        self.history = history_store

    def probe_all(self, session_name: str) -> list[ProbeResult]:
        """运行所有探活层级，返回结果列表"""
        s = self.store.get(session_name)
        if not s:
            return [ProbeResult(False, "state", 0, "Session not in local state")]

        results = [
            self._probe_rpc(s),
            self._probe_kernel_api(s),
        ]
        return results

    def _probe_rpc(self, s) -> ProbeResult:
        """层级 1: 调用 keep_alive_assignment RPC"""
        from colab_cli.client import Client, Prod
        import colab_cli.common as common

        t0 = time.time()
        try:
            common.state.client.keep_alive_assignment(s.endpoint)
            latency = (time.time() - t0) * 1000
            return ProbeResult(True, "rpc", latency)
        except Exception as e:
            latency = (time.time() - t0) * 1000
            from colab_cli.utils import get_status_code
            code = get_status_code(e)
            return ProbeResult(
                False, "rpc", latency,
                error=f"RPC failed: code={code}, {type(e).__name__}",
                detail={"status_code": code, "error_type": type(e).__name__}
            )

    def _probe_kernel_api(self, s) -> ProbeResult:
        """层级 2: 直接请求 Kernel API (HTTP GET)"""
        if not s.kernel_id:
            return ProbeResult(False, "kernel_api", 0, "No kernel_id")

        t0 = time.time()
        url = f"{s.url}/api/kernels/{s.kernel_id}"
        headers = {"X-Colab-Runtime-Proxy-Token": s.token}
        params = {"authuser": "0", "colab-runtime-proxy-token": s.token}

        try:
            r = requests.get(url, headers=headers, params=params, timeout=10)
            latency = (time.time() - t0) * 1000
            if r.status_code == 200:
                return ProbeResult(True, "kernel_api", latency)
            elif r.status_code in (401, 404):
                return ProbeResult(False, "kernel_api", latency,
                    error=f"Kernel API {r.status_code}")
            else:
                return ProbeResult(False, "kernel_api", latency,
                    error=f"Kernel API {r.status_code}")
        except requests.Timeout:
            latency = (time.time() - t0) * 1000
            return ProbeResult(False, "kernel_api", latency, error="Timeout")
        except requests.ConnectionError as e:
            latency = (time.time() - t0) * 1000
            return ProbeResult(False, "kernel_api", latency, error=f"ConnectionError: {e}")
```

### 4.3 Detect Layer — 故障分类与降噪

```python
"""session_detect.py — 故障诊断"""

from enum import Enum
from typing import Optional
from dataclasses import dataclass

class FailureMode(Enum):
    NO_FAILURE = "healthy"
    WS_DISCONNECT = "websocket_disconnect"      # WS 断开，但 HTTP RPC 可能还活着
    SESSION_PRUNED = "session_pruned"            # 服务端已清理
    GPU_QUOTA_EXCEEDED = "gpu_quota_exceeded"    # 创建时 GPU 配额不够
    NETWORK_ERROR = "network_error"              # 临时网络波动
    UNKNOWN = "unknown"

@dataclass
class DiagnosticResult:
    session_name: str
    failure_mode: FailureMode
    confidence: float        # 0.0 - 1.0
    probe_results: list      # 原始探活结果
    recommendation: str      # 建议操作


class SessionDiagnostic:
    """基于多层级探活结果进行故障诊断"""

    def diagnose(self, session_name: str, probes: list) -> DiagnosticResult:
        """分析探活结果，返回诊断结论"""

        # 1. 如果没有 session state，可能是被清理了
        # 调用者是上层，已经处理了这个 case

        # 2. 检查 RPC 探活结果
        rpc_result = next((p for p in probes if p.probe_type == "rpc"), None)
        kernel_result = next((p for p in probes if p.probe_type == "kernel_api"), None)

        # 情况 A: RPC 成功 + Kernel API 成功 = 健康
        if rpc_result and rpc_result.alive:
            if kernel_result and kernel_result.alive:
                return DiagnosticResult(
                    session_name, FailureMode.NO_FAILURE, 1.0, probes,
                    "Session healthy")

        # 情况 B: RPC 返回 404/401 = 会话被清理
        if rpc_result and rpc_result.detail.get("status_code") in (401, 404):
            return DiagnosticResult(
                session_name, FailureMode.SESSION_PRUNED, 0.95, probes,
                "Session no longer exists on server. Re-provision.")

        # 情况 C: RPC 网络错误 + Kernel API 也失败 = WS 断开或网络问题
        if rpc_result and not rpc_result.alive:
            net_errors = ["Timeout", "ConnectionError", "ConnectionRefused"]
            is_net_error = any(e in str(rpc_result.error or "") for e in net_errors)
            if is_net_error:
                # 区分是临时网络波动还是永久断连
                if kernel_result and not kernel_result.alive:
                    return DiagnosticResult(
                        session_name, FailureMode.WS_DISCONNECT, 0.7, probes,
                        "Kernel unreachable. Likely WS disconnect.")
                else:
                    return DiagnosticResult(
                        session_name, FailureMode.NETWORK_ERROR, 0.5, probes,
                        "Network error, will retry.")

        return DiagnosticResult(
            session_name, FailureMode.UNKNOWN, 0.3, probes,
            "Unclear state. Manual check recommended.")
```

### 4.4 Recover Layer — 自动恢复

```python
"""session_recover.py — 自动恢复策略"""

import subprocess
import os
import sys
import time
import json
from typing import Optional

ACCOUNT_CONFIGS = {
    "colab": {
        "home": os.path.expanduser("~"),
        "config_path": os.path.expanduser("~/.config/colab-cli/sessions.json"),
        "default": True,
    },
    "cb": {
        "home": os.path.expanduser("~/colab-accounts/account-cb"),
        "config_path": os.path.expanduser("~/colab-accounts/account-cb/.config/colab-cli/sessions.json"),
        "proxy": {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"},
        "no_proxy": "*.colab.dev,*.prod.colab.dev,localhost,127.0.0.1",
    },
    "cc": {
        "home": os.path.expanduser("~/colab-accounts/account-c"),
        "config_path": os.path.expanduser("~/colab-accounts/account-c/.config/colab-cli/sessions.json"),
        "proxy": {"http": "http://127.0.0.1:7890", "https": "http://127.0.0.1:7890"},
        "no_proxy": "*.colab.dev,*.prod.colab.dev,localhost,127.0.0.1",
    },
    "clb": {
        "home": os.path.expanduser("~/colab-accounts/clb"),
    },
}


class SessionRecoverer:
    """会话恢复器：检测到死亡后尝试在新的账号上重建"""

    def __init__(self, project_dir: str, script_name: str = "launch.py"):
        self.project_dir = project_dir
        self.script_name = script_name
        self.usage_log = os.path.expanduser("~/.config/colab-cli/shm_state.json")

    def _load_state(self) -> dict:
        if os.path.exists(self.usage_log):
            with open(self.usage_log) as f:
                return json.load(f)
        return {"account_rotation": [], "last_session": None}

    def _save_state(self, state: dict):
        os.makedirs(os.path.dirname(self.usage_log), exist_ok=True)
        with open(self.usage_log, "w") as f:
            json.dump(state, f, indent=2)

    def select_best_account(self) -> tuple[str, dict]:
        """
        选择当前最合适的账号：
        1. 优先选择没有活跃会话的账号
        2. 然后选择冷却时间最长的账号
        3. 如果有账号曾有成功使用的历史，优先
        """
        state = self._load_state()
        rotation = state.get("account_rotation", [])

        # 构建账号使用记录映射
        usage_map = {}
        for entry in rotation:
            alias = entry["account"]
            if alias not in usage_map or entry["timestamp"] > usage_map[alias]["timestamp"]:
                usage_map[alias] = entry

        # 对账号打分
        from datetime import datetime, timezone

        def score(alias: str, config: dict) -> float:
            """分数越低越优先"""
            base = 0.0
            if alias in usage_map:
                last = datetime.fromisoformat(usage_map[alias]["timestamp"])
                hours_since = (datetime.now(timezone.utc) - last).timestamp() / 3600
                # 刚用过的账号加分（不优先选择）
                if hours_since < 1:
                    base += 100
                elif hours_since < 6:
                    base += 50
                elif hours_since < 12:
                    base += 20
            # 默认账号优先
            if config.get("default"):
                base -= 10
            return base

        accounts = sorted(ACCOUNT_CONFIGS.items(), key=lambda x: score(x[0], x[1]))
        return accounts[0]

    def check_existing_sessions(self, account_alias: str) -> list:
        """检查指定账号的现有会话"""
        config = ACCOUNT_CONFIGS[account_alias]
        home = config["home"]

        env = os.environ.copy()
        env["HOME"] = home
        if "proxy" in config:
            env["HTTPS_PROXY"] = config["proxy"]["https"]
            env["HTTP_PROXY"] = config["proxy"]["http"]
        if "no_proxy" in config:
            env["no_proxy"] = config["no_proxy"]

        try:
            result = subprocess.run(
                ["colab", "sessions"],
                capture_output=True, text=True, timeout=30, env=env
            )
            if result.returncode == 0:
                lines = [l for l in result.stdout.split("\n")
                        if l.strip() and not l.startswith("[colab]")]
                return lines
        except subprocess.TimeoutExpired:
            pass
        return []

    def recover_session(self, dead_session_name: str,
                        accelerator: str = "T4",
                        source_dir: str = None) -> Optional[str]:
        """
        恢复会话：检测到死亡 → 选择下一个可用账号 → 创建新会话
        返回新会话名称，或 None 表示恢复失败
        """
        alias, config = self.select_best_account()
        home = config["home"]
        _suffix = f"{alias}-{int(time.time()) % 100000}"
        new_name = f"{dead_session_name}-{_suffix}"

        print(f"[shm] Recovering session '{dead_session_name}' "
              f"using account '{alias}' as '{new_name}'...")

        # 设置账号环境
        env = os.environ.copy()
        env["HOME"] = home
        if "proxy" in config:
            env["HTTPS_PROXY"] = config["proxy"]["https"]
            env["HTTP_PROXY"] = config["proxy"]["http"]
        if "no_proxy" in config:
            env["no_proxy"] = config["no_proxy"]

        # 创建新会话
        cmd = ["colab", "new", "--gpu", accelerator, "-s", new_name]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=120, env=env
            )
            if result.returncode != 0:
                print(f"[shm] Failed to create session: {result.stderr}")
                # 记录失败
                self._record_rotation(alias, new_name, success=False)
                return None
        except subprocess.TimeoutExpired:
            print(f"[shm] Timeout creating session")
            return None

        # 上传文件
        if source_dir and os.path.isdir(source_dir):
            for fname in os.listdir(source_dir):
                if fname.endswith(".py"):
                    local = os.path.join(source_dir, fname)
                    remote = f"/content/{fname}"
                    up_cmd = ["colab", "upload", "-s", new_name, local, remote]
                    try:
                        subprocess.run(up_cmd, capture_output=True, text=True,
                                    timeout=30, env=env)
                    except Exception:
                        pass

        # 记录成功
        self._record_rotation(alias, new_name, success=True)

        # 如果之前在运行训练，尝试恢复执行
        if source_dir and os.path.exists(os.path.join(source_dir, self.script_name)):
            subprocess.Popen(
                ["colab", "exec", "-s", new_name, "-f",
                 os.path.join(source_dir, self.script_name), "--timeout", "120"],
                env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

        print(f"[shm] Recovery session '{new_name}' created on account '{alias}'")
        return new_name

    def _record_rotation(self, alias: str, session_name: str, success: bool):
        from datetime import datetime, timezone
        state = self._load_state()
        state["account_rotation"].append({
            "account": alias,
            "session": session_name,
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        # 只保留最近 200 条
        state["account_rotation"] = state["account_rotation"][-200:]
        state["last_session"] = session_name
        self._save_state(state)
```

### 4.5 完整管理器

```python
"""session_health_manager.py — 主入口"""

import sys
import os
import time
import json
from datetime import datetime

sys.path.insert(0, os.path.expanduser(
    "~/.local/share/uv/tools/google-colab-cli/lib/python3.13/site-packages"))

def main():
    """每 2 分钟由 cron 调用一次"""
    from colab_cli.common import state
    from session_probe import SessionProbe
    from session_detect import SessionDiagnostic, FailureMode
    from session_recover import SessionRecoverer

    # 配置
    PROJECT_DIR = "/Users/mx/Desktop/projects/colab-cli/projects/transformer_iwslt"
    LAUNCH_SCRIPT = "launch.py"
    ENABLE_AUTO_RECOVER = True

    # 1. 获取所有本地会话
    sessions = state.store.list()
    if not sessions:
        print("[shm] No local sessions to monitor.")
        return

    # 2. 对每个会话进行探活
    probe = SessionProbe(state.store, state.history)
    diag = SessionDiagnostic()
    recoverer = SessionRecoverer(PROJECT_DIR, LAUNCH_SCRIPT)

    for name, s in sessions.items():
        print(f"[shm] Probing session '{name}'...")

        probes = probe.probe_all(name)
        result = diag.diagnose(name, probes)

        print(f"  → {result.failure_mode.value} (confidence={result.confidence})")

        # 3. 根据诊断结果采取行动
        if result.failure_mode in (FailureMode.SESSION_PRUNED,
                                    FailureMode.WS_DISCONNECT):

            # 确认会话状态（同步一次）
            _, assignments = state.sync_sessions()
            active_endpoints = {a.endpoint for a in assignments}
            if s.endpoint not in active_endpoints:
                print(f"[shm] Confirmed: session '{name}' is dead. "
                      f"Recovering...")
                if ENABLE_AUTO_RECOVER:
                    new_name = recoverer.recover_session(
                        name, accelerator=s.accelerator,
                        source_dir=PROJECT_DIR,
                    )
                    if new_name:
                        # 清理旧的失败会话
                        state.prune_session(name)
                        print(f"[shm] Recovery complete. New session: {new_name}")
                    else:
                        print(f"[shm] Recovery FAILED for '{name}'")
            else:
                print(f"[shm] Session '{name}' still on server. "
                      f"Possible network issue, will retry.")

if __name__ == "__main__":
    main()
```

### 4.6 Cron 集成

```bash
# 每 2 分钟运行一次健康检查
CronCreate \
  cron="*/2 * * * *" \
  prompt="""
Run: python3 /Users/mx/Desktop/projects/colab-cli/scripts/session_health_manager.py

Check the output. If it reports a dead session and created a recovery,
note the new session name and report it.
""" \
  durable=true \
  recurring=true

# MacOS 通知集成
osascript -e 'display notification "Session recovered: {new_name}" with title "Colab SHM"'
```

---

## 5. 多账号轮转策略

### 5.1 账号配置

```python
# 参考 CLAUDE.md 中的账号定义
ACCOUNTS = {
    "colab": {                       # 主账号 hackxie1998
        "home": "~",
        "proxy": None,               # 默认通过 Clash 代理
    },
    "cb": {                          # stefaniehu929
        "home": "~/colab-accounts/account-cb",
        "proxy": "http://127.0.0.1:7890",
        "no_proxy": "*.colab.dev,*.prod.colab.dev,localhost,127.0.0.1",
    },
    "cc": {                          # xbetterdetermine
        "home": "~/colab-accounts/account-c",
        "proxy": "http://127.0.0.1:7890",
        "no_proxy": "*.colab.dev,*.prod.colab.dev,localhost,127.0.0.1",
    },
    "clb": {                         # xieminghack
        "home": "~/colab-accounts/clb",
    },
}
```

### 5.2 轮转策略

```
轮转算法:

1. 检查所有账号的活跃会话数
   - 对每个账号: HOME=<account_home> colab sessions
   - 过滤出仍存活在服务端的会话

2. 选择目标账号的优先级:
   优先级 A: 当前有 0 个活跃会话的账号
   优先级 B: 距离上次使用时间最长的账号
   优先级 C: 默认账号 (colab)

3. 配额冷却跟踪:
   - 记录每次 GPU 分配的 timestamp
   - 建议冷却时间: 至少 6 小时
   - 如果所有账号都在冷却中 → 等待 + 通知

4. 轮转状态文件 (~/.config/colab-cli/rotation_state.json):
   {
     "current_account": "cc",
     "last_rotation": "2026-06-11T15:00:00Z",
     "account_history": [
       {"account": "colab", "used_at": "...", "sessions_created": 3},
       {"account": "cb",    "used_at": "...", "sessions_created": 2},
     ],
     "cooldowns": {
       "colab": {"expires_at": "2026-06-12T03:00:00Z"},
       "cb":    {"expires_at": "2026-06-11T22:00:00Z"},
       "cc":    {"expires_at": null},
       "clb":   {"expires_at": null}
     }
   }
```

### 5.3 Shell 封装

```bash
#!/bin/bash
# rotate-colab.sh — 跨账号操作封装

ACCOUNT=${1:-colab}
shift 2>/dev/null

case "$ACCOUNT" in
  colab)
    HOME=~ colab "$@"
    ;;
  cb)
    HOME=~/colab-accounts/account-cb \
    HTTPS_PROXY=http://127.0.0.1:7890 \
    HTTP_PROXY=http://127.0.0.1:7890 \
    no_proxy="*.colab.dev,*.prod.colab.dev,localhost,127.0.0.1" \
    colab "$@"
    ;;
  cc)
    HOME=~/colab-accounts/account-c \
    HTTPS_PROXY=http://127.0.0.1:7890 \
    HTTP_PROXY=http://127.0.0.1:7890 \
    no_proxy="*.colab.dev,*.prod.colab.dev,localhost,127.0.0.1" \
    colab "$@"
    ;;
  clb)
    HOME=~/colab-accounts/clb colab "$@"
    ;;
  *)
    echo "Unknown account: $ACCOUNT"
    echo "Usage: $0 {colab|cb|cc|clb} <colab-args>"
    exit 1
    ;;
esac
```

### 5.4 自动轮转脚本

```bash
#!/bin/bash
# auto-assign.sh — 自动在可用账号上创建会话

SESSION_NAME=${1:-"auto-$(date +%s)"}
GPU_TYPE=${2:-T4}

for account in cc cb clb colab; do
  echo "[auto-assign] Trying account: $account"

  # 检查该账号现有会话数
  if [ "$account" = "colab" ]; then
    SESSIONS=$(colab sessions 2>/dev/null | grep -vc "^\[colab\]")
  else
    HOME=~/colab-accounts/account-$account \
    HTTPS_PROXY=http://127.0.0.1:7890 \
    HTTP_PROXY=http://127.0.0.1:7890 \
    no_proxy="*.colab.dev,*.prod.colab.dev,localhost,127.0.0.1" \
    colab sessions 2>/dev/null | grep -vc "^\[colab\]"
  fi

  # 如果已有活跃会话，跳过 (每账号限 1 GPU)
  if [ "$SESSIONS" -gt 0 ] 2>/dev/null; then
    echo "  -> Account $account has $SESSIONS active session(s), skipping"
    continue
  fi

  # 尝试创建
  echo "  -> Creating session on $account..."
  if [ "$account" = "colab" ]; then
    colab new --gpu "$GPU_TYPE" -s "$SESSION_NAME" 2>&1
  else
    HOME=~/colab-accounts/account-$account \
    HTTPS_PROXY=http://127.0.0.1:7890 \
    HTTP_PROXY=http://127.0.0.1:7890 \
    no_proxy="*.colab.dev,*.prod.colab.dev,localhost,127.0.0.1" \
    colab new --gpu "$GPU_TYPE" -s "$SESSION_NAME" 2>&1
  fi

  if [ $? -eq 0 ]; then
    echo "[auto-assign] Session created on $account: $SESSION_NAME"
    exit 0
  fi
  echo "  -> Failed on $account, trying next..."
done

echo "[auto-assign] ALL ACCOUNTS EXHAUSTED"
exit 1
```

### 5.5 配额估算模型

```python
"""quota_model.py — GPU 配额预测"""

from datetime import datetime, timedelta, timezone
from typing import Dict, Optional
import json
import os

class QuotaModel:
    """
    基于经验的配额估算模型。

    Google 不公开配额算法，但根据实际使用模式可以建立经验模型：
    - 免费账号每天约 10-12 小时 GPU
    - 冷却期 12-24 小时（重度使用后可能更长）
    - 并行使用多个短 session（<4min）消耗更快
    """

    def __init__(self, state_path: str = "~/.config/colab-cli/shm_state.json"):
        self.path = os.path.expanduser(state_path)
        self.state = self._load()

    def _load(self) -> dict:
        if os.path.exists(self.path):
            with open(self.path) as f:
                return json.load(f)
        return {"account_rotation": []}

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(self.state, f, indent=2)

    def record_usage(self, account: str, gpu_type: str, duration_min: float):
        """记录一次 GPU 使用"""
        entry = {
            "account": account,
            "gpu_type": gpu_type,
            "duration_min": duration_min,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.state.setdefault("account_rotation", []).append(entry)
        # 只保留最近 7 天
        cutoff = datetime.now(timezone.utc) - timedelta(days=7)
        self.state["account_rotation"] = [
            e for e in self.state["account_rotation"]
            if datetime.fromisoformat(e["timestamp"]) > cutoff
        ]
        self._save()

    def get_account_health(self, account: str) -> Dict:
        """评估账号健康度"""
        usage = [e for e in self.state.get("account_rotation", [])
                if e["account"] == account]

        if not usage:
            return {"status": "fresh", "reason": "No recent usage"}

        last = datetime.fromisoformat(usage[-1]["timestamp"])
        hours_since = (datetime.now(timezone.utc) - last).total_seconds() / 3600

        # 最近 24 小时使用总时长
        cutoff_24h = datetime.now(timezone.utc) - timedelta(hours=24)
        recent_24h = sum(
            e["duration_min"] for e in usage
            if datetime.fromisoformat(e["timestamp"]) > cutoff_24h
        )

        if recent_24h > 180:          # > 3 小时 / 24h
            return {"status": "depleted", "reason": "Heavy usage (>3hr in 24h)"}
        elif hours_since < 1:
            return {"status": "cooling", "reason": f"Last used {hours_since:.0f}h ago"}
        else:
            return {"status": "available", "reason": f"Last used {hours_since:.0f}h ago"}

    def best_account(self, available_accounts: list) -> Optional[str]:
        """返回当前最优账号"""
        scores = []
        for acct in available_accounts:
            health = self.get_account_health(acct)
            if health["status"] == "depleted":
                scores.append((acct, 999))
            elif health["status"] == "cooling":
                scores.append((acct, 50))
            else:
                scores.append((acct, 0))
        scores.sort(key=lambda x: x[1])
        return scores[0][0] if scores else None
```

---

## 6. 实施优先级

### P0 — 紧急修复（影响训练的致命问题）

| 编号 | 任务 | 影响 | 工作量 |
|------|------|------|-------|
| P0.1 | **Colab exec 超时兜底**：当前 `--timeout` 默认 10s，训练脚本稍长就会超时。需要改为 `--timeout 120` 或更高 | 高：导致所有 check_progress 调用频繁失败 | 5min |
| P0.2 | **WebSocket 断线检测**：在 check_progress.py 中增加 colab 会话级别的健康检测，不只看进程 | 高：当前误报很多 | 1h |
| P0.3 | **check_progress 结果通知**：CRITICAL 告警通过 osascript 发送 macOS 通知 | 高：训练失败无人知晓 | 30min |

**P0.1 实现：**
```bash
# 将 cron 中的 exec 命令改为使用更长超时
colab exec -s <session> -f check_progress.py --timeout 120
```

**P0.2 实现（在 check_progress.py 中增加）：**
```python
def check_colab_session(session_name):
    """额外检查 colab 会话本身是否存活"""
    try:
        result = subprocess.run(
            ["colab", "status", "-s", session_name],
            capture_output=True, text=True, timeout=15
        )
        if "not found" in result.stdout.lower():
            print("[check] CRITICAL: Colab session is DEAD (not found)")
            return False
        return True
    except Exception as e:
        print(f"[check] WARNING: Could not check session status: {e}")
        return None
```

### P1 — 高优先级（减少训练中断）

| 编号 | 任务 | 影响 | 工作量 |
|------|------|------|-------|
| P1.1 | **Keep-alive daemon 增强**：当前 daemon 连续 2 次 4xx 就退出。改为记录 daemon 退出原因到单独的文件，供监控读取 | 中：目前 daemon 退出无声无息 | 1h |
| P1.2 | **watchdog 改进**：watchdog 除了写心跳，还应该检测自己能否 ping 通 colab API，如果断连则尝试重新连接 | 中：心跳只证明 VM 活着，不证明可访问 | 2h |
| P1.3 | **GPU 使用日志自动管理**：每次创建/停止会话时记录 GPU 使用时间，帮助判断配额状态 | 中：手动操作容易忘记哪个账号还能用 | 1h |
| P1.4 | **colab exec --timeout 自动适配**：根据脚本类型自动选择超时（check_progress=120s, train=3600s） | 低但实用 | 30min |

### P2 — 中期（自动恢复基建）

| 编号 | 任务 | 影响 | 工作量 |
|------|------|------|-------|
| P2.1 | **Session Health Manager 基础版本**：实现 Python 脚本，每 2 分钟探活所有会话 | 高：自动恢复的前置条件 | 4h |
| P2.2 | **多账号自动轮转**：实现 `auto-assign.sh`，支持在账号间自动切换 | 高：充分利用 4 账号 | 2h |
| P2.3 | **配额冷却模型**：跟踪每个账号的 GPU 使用量，预测可用性 | 中：减少分配失败 | 2h |
| P2.4 | **自动恢复 + 断点续训**：会话死亡检测后，自动在新账号创建会话、上传文件、启动带 `--resume` 的训练 | 高：全自动修复 | 4h |

### P3 — 长期优化

| 编号 | 任务 | 影响 | 工作量 |
|------|------|------|-------|
| P3.1 | **会话池管理器**：维护 N 个健康会话跨 M 个账号，自动补充死亡会话 | 中：复杂的全局调度 | 8h |
| P3.2 | **主动预分配**：在 GPU 配额可能用完前，提前在另一个账号分配 | 低：预测准确度有限 | 4h |
| P3.3 | **CI 集成**：与本地开发流程集成，自动管理训练会话 | 低：锦上添花 | 6h |
| P3.4 | **Web 仪表盘**：简单的 Web GUI 查看所有会话状态 | 低：辅助功能 | 8h |

### 6.1 实施路线图

```
Week 1 (P0):
  ┌─────────────────────────────────────────────┐
  │ P0.1: exec 超时调整     [5min]              │
  │ P0.2: 会话层健康检测     [1h]                │
  │ P0.3: macOS 通知集成    [30min]              │
  └─────────────────────────────────────────────┘

Week 2 (P1):
  ┌─────────────────────────────────────────────┐
  │ P1.1: daemon 退出原因持久化  [1h]           │
  │ P1.2: watchdog 网络探活     [2h]             │
  │ P1.3: GPU 使用日志          [1h]             │
  └─────────────────────────────────────────────┘

Week 3-4 (P2):
  ┌─────────────────────────────────────────────┐
  │ P2.1: Session Health Manager   [4h]         │
  │ P2.2: 多账号自动轮转           [2h]          │
  │ P2.3: 配额冷却模型             [2h]          │
  │ P2.4: 自动恢复+断点续训        [4h]          │
  └─────────────────────────────────────────────┘
```

---

## 附录 A: 相关文件路径

```
# colab CLI 源码
~/.local/share/uv/tools/google-colab-cli/lib/python3.13/site-packages/colab_cli/
├── cli.py                    # CLI 入口，注册所有子命令
├── common.py                 # State 单例 (store, client, history)
├── state.py                  # SessionState, StateStore (sessions.json)
├── client.py                 # API 客户端 (assign, keep_alive, list_assignments)
├── runtime.py                # ColabRuntime (kernel连接、代码执行、WS hook)
├── utils.py                  # get_status_code, is_terminal_error
├── history.py                # HistoryLogger (事件记录)
├── contents.py               # ContentsClient (文件上传/下载)
├── commands/
│   ├── session.py            # new, status, sessions, stop, keep_alive daemon
│   ├── execution.py          # exec, repl, console
│   ├── run.py                # run (一站式 new + exec + stop)
│   ├── files.py              # upload, download, ls, rm, edit
│   └── automation.py         # auth, drivemount, install

# Session 状态
~/.config/colab-cli/sessions.json

# History 日志
~/.config/colab-cli/history/<session_name>.jsonl

# 项目监控文件 (VM 端)
/content/heartbeat.json       # watchdog.py 写入
/content/metrics.jsonl         # train.py 写入
/content/train.log            # train.py stdout/stderr

# 项目监控脚本 (本地)
projects/alexnet_imagenette/watchdog.py
projects/alexnet_imagenette/check_progress.py
projects/transformer_iwslt/check_progress.py
projects/nanogpt/check_progress.py
projects/vllm-compare/check_progress.py
```

## 附录 B: Daemon 退出原因速查表

| 退出原因 | 含义 | 可能原因 | 处理建议 |
|---------|------|---------|---------|
| `time_limit_reached` | daemon 运行满 24 小时 | 正常上限 | 需要新建会话 |
| `session_not_found` | local state 中 session 被删 | 1. `sync_sessions()` 同步清理<br>2. 其他进程调用了 `prune_session` | 检查同步日志 |
| `endpoint_mismatch` | endpoint 已变更 | 被同一 name 新建会话覆盖 | 检查是否有并发操作 |
| `consecutive_4xx_errors` | 连续 2 次 4xx RPC 失败 | 1. 会话被服务端清理<br>2. OAuth token 过期 | 会话大概率已死 |

## 附录 C: 代理问题排查命令

```bash
# 测试直连 (colab 主账号)
colab status -s <name>

# 测试通过代理 (cb/cc 账号)
HOME=~/colab-accounts/account-c \
HTTPS_PROXY=http://127.0.0.1:7890 \
HTTP_PROXY=http://127.0.0.1:7890 \
no_proxy="*.colab.dev,*.prod.colab.dev,localhost,127.0.0.1" \
colab status -s <name>

# 测试 WebSocket 连接 (使用 curl 升级到 WS)
# 注意: curl 7.86+ 支持 WebSocket
curl -v wss://<url>/api/kernels/<kernel_id>/channels \
  -H "X-Colab-Runtime-Proxy-Token: <token>" \
  -H "X-Colab-Client-Agent: colab-cli" \
  --http1.1 \
  --connect-timeout 10
```

## 附录 D: 关键代码分析

### D.1 `is_terminal_error` — 终端错误检测

```python
# utils.py:34-44
def is_terminal_error(e: Exception) -> bool:
    """Checks if an exception indicates a lost session (404/401)."""
    code = get_status_code(e)
    if code in (404, 401):
        return True
    err_msg = str(e)
    if "404" in err_msg or "401" in err_msg:
        return True
    return False
```

**分析：** 这是当前唯一用于判断"会话是否死亡"的逻辑。使用场景在 `execution.py` 中 try-catch wrapper 里。局限是：
- 只判断 401/404，不处理 400 (GPU quota), 403 (OAuth scope), 502 (gateway)
- 不区分网络超时和真正的会话死亡

### D.2 `sync_sessions` — 会话同步与清理

```python
# common.py:81-112
def sync_sessions(self):
    # ...
    assignments = self.client.list_assignments()
    active_endpoints = {a.endpoint for a in assignments}
    for name, s in list(self._sessions.items()):
        if s.endpoint not in active_endpoints:
            self.prune_session(name)
```

**分析：** 这是"静默清理"的关键路径——当调用 `colab sessions` 或 `colab status` 时，如果服务端不再知道某个 session，它会被**无声删除**（只是打印一行 `Pruned N stale local session(s)`）。这意味着：
- 用户可能刚发现会话消失，其实是之前的 commands 触发了清理
- Daemon 随后会因为 `session_not_found` 退出

### D.3 `keep_alive` — 保活守护进程

```python
# session.py:424-502
while time.time() - start_time < max_duration:
    s = state.store.get(session_name)
    if not s:
        reason = "session_not_found"
        break
    try:
        state.client.keep_alive_assignment(endpoint)
        consecutive_4xx = 0
    except Exception as e:
        code = get_status_code(e)
        if code is not None and 400 <= code < 500:
            consecutive_4xx += 1
            if consecutive_4xx >= 2:
                reason = "consecutive_4xx_errors"
                break
    time.sleep(60)
```

**分析：** Daemon 的核心局限：
- **60s 间隔** — 从会话死亡到 daemon 发现并退出最多需要 ~2 分钟
- **不处理网络错误** — 非 4xx 异常（如超时、连接拒绝）会被忽略，继续重试
- **不写退出原因到文件** — 退出原因只记录到 history JSONL，监控脚本需要额外解析
- **不知服务端状态** — 只通过 RPC 返回码判断，不通过 `list_assignments` 确认

---

*文档版本: v1.0 | 生成日期: 2026-06-11*
