import subprocess, os

print("=== Process Status ===")
r = subprocess.run(["pgrep", "-a", "python"], capture_output=True, text=True)
print(r.stdout if r.stdout else "No python process running")

print("=== Log Tail ===")
log_path = "/content/cuda_tutorial.log"
if os.path.exists(log_path):
    with open(log_path) as f:
        lines = f.readlines()
        for line in lines[-50:]:
            print(line, end="")
    print(f"\n--- {len(lines)} total lines ---")
else:
    print("Log file not found")
