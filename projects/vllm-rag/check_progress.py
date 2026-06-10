"""Check RAG pipeline progress on Colab VM."""
import os, json, subprocess

server_log = "/content/vllm_server.log"
eval_log = "/content/rag_eval.log"
results_file = "/content/rag_results.json"

# Check server process
print("── vLLM Server ──")
srv = subprocess.run(
    ["pgrep", "-f", "server.py"], capture_output=True, text=True
)
print(f"  Server: {'RUNNING' if srv.stdout.strip() else 'NOT RUNNING'} (PID: {srv.stdout.strip() or 'none'})")

# Check eval process
evl = subprocess.run(
    ["pgrep", "-f", "rag_eval.py"], capture_output=True, text=True
)
print(f"  Eval:   {'RUNNING' if evl.stdout.strip() else 'NOT RUNNING'} (PID: {evl.stdout.strip() or 'none'})")

# Latest eval log lines
print("\n── Last 15 Eval Log Lines ──")
if os.path.exists(eval_log):
    with open(eval_log) as f:
        lines = f.readlines()
        for line in lines[-15:]:
            print(f"  {line.rstrip()}")
else:
    print("  (no eval log yet)")

# Results summary
print("\n── Results ──")
if os.path.exists(results_file):
    with open(results_file) as f:
        data = json.load(f)
    rm = data.get("retrieval_metrics", {})
    gm = data.get("generation_metrics", {})
    sm = data.get("system_metrics", {})
    print(f"  Queries: {len(data.get('per_question', []))}")
    print(f"  Recall@1: {rm.get('recall_at_1', '?')}")
    print(f"  Recall@3: {rm.get('recall_at_3', '?')}")
    print(f"  MRR: {rm.get('mrr', '?')}")
    print(f"  Exact Match: {gm.get('exact_match', '?')}")
    print(f"  Token F1: {gm.get('token_f1_avg', '?')}")
    print(f"  Avg Latency: {sm.get('avg_latency_s', '?')}s")
    print(f"  Peak VRAM: {sm.get('peak_vram_gb', '?')}GB")
else:
    print("  (no results yet)")
