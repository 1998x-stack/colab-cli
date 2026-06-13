"""Chain-of-Thought strategy: single prompt, single vLLM batch call, parse answers."""
import time
from vllm import SamplingParams
from prompts import make_cot_prompt, extract_final_answer


def run_cot(llm, examples: list[dict], max_tokens: int = 512) -> list[dict]:
    """Run CoT on all examples in a single batched vLLM call.

    Returns list of result dicts with prediction and token count.
    """
    prompts = []
    for ex in examples:
        prompts.append(make_cot_prompt(ex["question"], ex["context"]))

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
        stop=["\n\nContext:", "\n\nQuestion:"],
    )

    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    total_time = time.time() - t0

    results = []
    for ex, out in zip(examples, outputs):
        raw = out.outputs[0].text
        predicted = extract_final_answer(raw) or raw.strip()
        n_tokens = len(out.outputs[0].token_ids)

        results.append({
            "id": ex["id"],
            "question": ex["question"],
            "answer": ex["answer"],
            "prediction": predicted,
            "raw_output": raw,
            "tokens": n_tokens,
        })

    amortized = total_time / len(examples) if examples else 0
    print(f"CoT: {len(examples)} examples in {total_time:.1f}s "
          f"({amortized:.2f}s/example amortized)")

    return results
