"""Bootstrap clash-meta proxy on a Colab VM using proxy.yaml.

    colab upload proxy.yaml /content/proxy.yaml
    colab exec -f vm-proxy-bootstrap.py --timeout 60

If SS servers in proxy.yaml are reachable from the VM, starts clash on port 7890.
If unreachable (common from GCP), reports direct connectivity status and
exits cleanly — Colab VMs have excellent direct internet access to most services.

After bootstrap, set these env vars in subprocesses if proxy is active:

    HTTPS_PROXY=http://127.0.0.1:7890
    HTTP_PROXY=http://127.0.0.1:7890
    ALL_PROXY=socks5://127.0.0.1:7890
"""

import subprocess
import sys
import os
import time

MIHOMO_VER = os.environ.get("MIHOMO_VER", "1.19.0")
MIHOMO_URL = (
    f"https://github.com/MetaCubeX/mihomo/releases/download/"
    f"v{MIHOMO_VER}/mihomo-linux-amd64-v{MIHOMO_VER}.gz"
)
MIHOMO_PATH = "/usr/local/bin/mihomo"
CONFIG_SRC = "/content/proxy.yaml"
CONFIG_DIR = "/etc/mihomo"
CONFIG_DST = f"{CONFIG_DIR}/config.yaml"
MIXED_PORT = 7890

# Key services to test
TEST_URLS = [
    ("https://www.google.com", "Google"),
    ("https://huggingface.co", "HuggingFace"),
    ("https://pypi.org", "PyPI"),
    ("https://github.com", "GitHub"),
]


def run(cmd, **kwargs):
    print(f"[proxy] + {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    subprocess.run(cmd, check=True, **kwargs)


def check_direct():
    """Quick check of direct connectivity from the VM."""
    all_ok = True
    for url, label in TEST_URLS:
        result = subprocess.run(
            ["curl", "-s", "--max-time", "5", url,
             "-o", "/dev/null", "-w", "%{http_code}"],
            capture_output=True, text=True,
        )
        code = result.stdout.strip()
        ok = code == "200"
        if not ok:
            all_ok = False
        print(f"[proxy] Direct: {label:20s} → HTTP {code} {'  OK' if ok else '  BLOCKED'}")
    return all_ok


def check_ss_reachable():
    """Test if at least one SS server in proxy.yaml is reachable."""
    # Extract host:port pairs from proxy.yaml
    import re
    if not os.path.exists(CONFIG_SRC):
        return False
    with open(CONFIG_SRC) as f:
        content = f.read()
    # Match: server: bit-XX.kunlun03dns.com, port: 12xxx
    servers = set()
    for m in re.finditer(r"server:\s*(\S+).*?port:\s*(\d+)", content):
        host, port = m.group(1), m.group(2)
        servers.add((host, port))

    reachable = False
    for host, port in sorted(servers)[:6]:  # Test up to 6
        result = subprocess.run(
            ["timeout", "3", "bash", "-c",
             f"echo >/dev/tcp/{host}/{port} 2>/dev/null && echo OPEN || echo FAIL"],
            capture_output=True, text=True,
        )
        status = "OPEN" if "OPEN" in result.stdout else "TIMEOUT"
        if status == "OPEN":
            reachable = True
        print(f"[proxy] SS server {host}:{port} → {status}")

    return reachable


def install_mihomo():
    if os.path.exists(MIHOMO_PATH) and subprocess.run(
        [MIHOMO_PATH, "-v"], capture_output=True
    ).returncode == 0:
        print("[proxy] mihomo already installed, skipping download.")
        return

    print(f"[proxy] Downloading mihomo v{MIHOMO_VER} ...")
    run(["wget", "-q", "--show-progress", "-O", "/tmp/mihomo.gz", MIHOMO_URL])
    run(["gunzip", "-f", "/tmp/mihomo.gz"])
    run(["install", "-m", "755", "/tmp/mihomo", MIHOMO_PATH])
    print("[proxy] mihomo installed.")


def start_clash():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    run(["cp", CONFIG_SRC, CONFIG_DST])

    subprocess.run(["pkill", "-f", "mihomo"], capture_output=True)
    time.sleep(1)

    log = "/tmp/mihomo.log"
    with open(log, "w") as f:
        proc = subprocess.Popen(
            [MIHOMO_PATH, "-d", CONFIG_DIR],
            stdout=f, stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    # Wait for SS connections to establish
    time.sleep(8)

    for attempt in range(8):
        result = subprocess.run(
            ["curl", "-s", "--max-time", "5",
             "-x", f"http://127.0.0.1:{MIXED_PORT}",
             "https://www.google.com", "-o", "/dev/null", "-w", "%{http_code}"],
            capture_output=True, text=True,
        )
        code = result.stdout.strip()
        if code == "200":
            print(f"[proxy] OK. PID={proc.pid}  port={MIXED_PORT}  log={log}")
            return True
        print(f"[proxy] Waiting for proxy... (attempt {attempt + 1}/8, HTTP {code})")
        time.sleep(3)

    print("[proxy] Proxy test failed. Mihomo log tail:")
    subprocess.run(["tail", "-30", log])
    return False


if __name__ == "__main__":
    # 1. Check direct connectivity first
    print("[proxy] === Direct VM connectivity ===")
    direct_ok = check_direct()
    print()

    # 2. Check if SS servers are reachable
    if not os.path.exists(CONFIG_SRC):
        print(f"[proxy] ERROR: {CONFIG_SRC} not found. Upload proxy.yaml first:")
        print("  colab upload proxy.yaml /content/proxy.yaml")
        sys.exit(1)

    print("[proxy] === SS server reachability ===")
    ss_ok = check_ss_reachable()
    print()

    if not ss_ok:
        print("[proxy] RESULT: SS servers unreachable from this VM.")
        print("[proxy] Colab VMs have excellent direct internet from GCP.")
        if direct_ok:
            print("[proxy] All key services accessible directly — no proxy needed.")
        else:
            print("[proxy] Some services blocked. A different proxy provider is needed.")
        print("[proxy] Set proxy env vars ONLY if a working proxy is available:")
        print("  HTTPS_PROXY=http://127.0.0.1:7890")
        print("  HTTP_PROXY=http://127.0.0.1:7890")
        print("  ALL_PROXY=socks5://127.0.0.1:7890")
        sys.exit(0)

    # 3. SS reachable — install and start clash
    print("[proxy] === Starting clash-meta proxy ===")
    install_mihomo()
    if start_clash():
        print("\n[proxy] Proxy active. Use in subprocess env:")
        print("  HTTPS_PROXY=http://127.0.0.1:7890")
        print("  HTTP_PROXY=http://127.0.0.1:7890")
        print("  ALL_PROXY=socks5://127.0.0.1:7890")
    else:
        print("\n[proxy] FAILED to start proxy. Use direct connections.")
        sys.exit(1)
