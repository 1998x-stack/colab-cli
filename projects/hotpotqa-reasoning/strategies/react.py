"""ReAct strategy: multi-turn batched loop. Each step generates one turn for all
active examples. Finished examples drop out. Converges when all hit Final Answer
or max 5 steps."""
import time
from vllm import SamplingParams
from prompts import (
    make_react_initial_prompt,
    make_react_continue_prompt,
    extract_final_answer,
)

MAX_STEPS = 5
MAX_TOKENS_PER_STEP = 256


def run_react(llm, examples: list[dict]) -> list[dict]:
    """Run ReAct on all examples with batched multi-turn generation.

    Each step sends one prompt per still-active example in a single vLLM
    call. Examples that produce 'Final Answer:' are removed from the
    active set for subsequent steps.
    """
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_TOKENS_PER_STEP,
        stop=["\n\nContext:", "\n\nQuestion:"],
    )

    states = {
        ex["id"]: {
            "example": ex,
            "history": "",
            "prediction": None,
            "steps": 0,
            "total_tokens": 0,
            "total_latency_s": 0.0,
            "done": False,
        }
        for ex in examples
    }

    for step in range(1, MAX_STEPS + 1):
        active_ids = [eid for eid, s in states.items() if not s["done"]]
        if not active_ids:
            break

        prompts = []
        for eid in active_ids:
            s = states[eid]
            ex = s["example"]
            if step == 1:
                prompts.append(make_react_initial_prompt(ex["question"], ex["context"]))
            else:
                prompts.append(make_react_continue_prompt(
                    ex["question"], ex["context"], s["history"]
                ))

        t0 = time.time()
        outputs = llm.generate(prompts, sampling_params)
        step_time = time.time() - t0

        for eid, out in zip(active_ids, outputs):
            s = states[eid]
            raw = out.outputs[0].text
            n_tokens = len(out.outputs[0].token_ids)
            ttft = _react_ttft(out)

            s["total_tokens"] += n_tokens
            if ttft is not None:
                s["total_latency_s"] += ttft
            s["steps"] = step

            answer = extract_final_answer(raw)
            if answer is not None:
                s["prediction"] = answer
                s["done"] = True

            s["history"] += f"Thought:{raw}\n"

        n_done = sum(1 for s in states.values() if s["done"])
        print(f"ReAct step {step}: {len(active_ids)} active, {n_done}/{len(examples)} done "
              f"({step_time:.1f}s)")

    results = []
    for s in states.values():
        ex = s["example"]
        if s["prediction"] is None:
            s["prediction"] = _force_extract(s["history"])

        results.append({
            "id": ex["id"],
            "question": ex["question"],
            "answer": ex["answer"],
            "prediction": s["prediction"],
            "raw_output": s["history"],
            "latency_s": round(s["total_latency_s"], 3),
            "tokens": s["total_tokens"],
            "steps": s["steps"],
        })

    total_steps = sum(r["steps"] for r in results)
    avg_steps = total_steps / len(results) if results else 0
    print(f"ReAct done: avg {avg_steps:.1f} steps/example, "
          f"{sum(r['tokens'] for r in results)} total tokens")

    return results


def _react_ttft(output) -> float | None:
    m = output.metrics
    if m is not None and m.first_token_time is not None and m.arrival_time is not None:
        return m.first_token_time - m.arrival_time
    return None


def _force_extract(history: str) -> str:
    """Last-resort extraction: take the last non-empty line."""
    answer = extract_final_answer(history)
    if answer:
        return answer
    for line in reversed(history.strip().splitlines()):
        line = line.strip()
        if line:
            return line
    return ""
