"""Test basic multiprocessing on Colab."""
import multiprocessing as mp
import torch

def worker(q):
    q.put(f"GPU: {torch.cuda.get_device_name(0)}")

print(f"Start method: {mp.get_start_method()}")
q = mp.Queue()
p = mp.Process(target=worker, args=(q,))
p.start()
p.join(timeout=30)
if p.is_alive():
    print("FAIL: process still alive after 30s")
    p.terminate()
else:
    print(f"OK: {q.get()}")
