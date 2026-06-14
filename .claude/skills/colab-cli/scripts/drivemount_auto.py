#!/usr/bin/env python3
"""Wrapper around colab drivemount that auto-opens auth URL in browser."""
import subprocess, re, sys, os, time, threading

os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7890")
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7890")
os.environ.setdefault("ALL_PROXY", "socks5://127.0.0.1:7890")

args = sys.argv[1:] if len(sys.argv) > 1 else []
cmd = ["/Users/mx/.local/bin/colab", "drivemount"] + args

print(f"[wrapper] Running: {' '.join(cmd)}")

proc = subprocess.Popen(
    cmd,
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1,
)

url_pattern = re.compile(r'https://accounts\.google\.com/o/oauth2/[^\s]+')
auth_url = None
output_lock = threading.Lock()
output_done = threading.Event()

def reader_thread():
    """Read stdout and print it. Sets auth_url when detected."""
    global auth_url
    for line in iter(proc.stdout.readline, ""):
        with output_lock:
            sys.stdout.write(line)
            sys.stdout.flush()
        if auth_url is None:
            match = url_pattern.search(line)
            if match:
                auth_url = match.group(0).rstrip(".")
    output_done.set()

reader = threading.Thread(target=reader_thread, daemon=True)
reader.start()

# Wait for auth URL (max 60s)
start = time.time()
while auth_url is None and reader.is_alive() and (time.time() - start) < 60:
    time.sleep(0.5)

if auth_url:
    print(f"\n[wrapper] Opening browser for OAuth...")
    subprocess.run(["open", auth_url], timeout=5)
    print("[wrapper] Complete authorization in your browser (180s timeout)...")
    sys.stdout.flush()
    time.sleep(180)
    print("[wrapper] Sending Enter to continue...")
    sys.stdout.flush()
    try:
        proc.stdin.write("\n")
        proc.stdin.flush()
    except (BrokenPipeError, OSError):
        pass
else:
    print("[wrapper] No auth URL detected. Sending Enter anyway...")
    sys.stdout.flush()
    try:
        proc.stdin.write("\n")
        proc.stdin.flush()
    except (BrokenPipeError, OSError):
        pass

# Wait for reader to finish
reader.join(timeout=30)
proc.wait()
print(f"\n[wrapper] Exit: {proc.returncode}")
