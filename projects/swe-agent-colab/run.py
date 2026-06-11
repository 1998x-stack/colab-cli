#!/usr/bin/env python3
"""Run SWE-agent on all tasks and collect metrics."""

import json
import logging
import sys
import time
from pathlib import Path

from agent import Agent
from environment import Environment
from tasks import get_tasks

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("agent.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("run")


def main(output_dir: str = "output"):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    tasks = get_tasks()
    logger.info(f"Running {len(tasks)} tasks")

    all_metrics = []

    for task in tasks:
        logger.info(f"=== Task: {task.task_id} ===")

        env = Environment(repo_url=task.repo_url, commit=task.commit)
        logger.info(f"Setting up environment: cloning {task.repo_url}")
        env.setup()

        agent = Agent()
        logger.info("Starting agent run")
        t0 = time.perf_counter()

        result = agent.run(
            env=env,
            problem_statement=task.problem_statement,
            test_cmd=task.test_cmd,
        )

        elapsed = time.perf_counter() - t0

        metrics = {
            "task_id": task.task_id,
            "resolved": result.resolved,
            "steps_taken": result.steps_taken,
            "total_tokens": result.model_stats["prompt_tokens"] + result.model_stats["completion_tokens"],
            "prompt_tokens": result.model_stats["prompt_tokens"],
            "completion_tokens": result.model_stats["completion_tokens"],
            "model_queries": result.model_stats["total_queries"],
            "total_time_seconds": elapsed,
            "model_time_seconds": result.model_stats["total_time"],
            "errors": result.errors,
        }
        all_metrics.append(metrics)
        logger.info(f"Result: resolved={result.resolved}, steps={result.steps_taken}, "
                     f"tokens={metrics['total_tokens']}, time={elapsed:.1f}s")

        # Save trajectory
        traj_path = out / f"trajectory_{task.task_id}.json"
        traj_path.write_text(json.dumps({
            "task_id": task.task_id,
            "resolved": result.resolved,
            "patch": result.patch,
            "trajectory": result.trajectory,
            "metrics": metrics,
        }, indent=2))
        logger.info(f"Trajectory saved to {traj_path}")

    # Save aggregate metrics
    metrics_path = out / "metrics.json"
    summary = {
        "total_tasks": len(tasks),
        "resolved_count": sum(1 for m in all_metrics if m["resolved"]),
        "pass_rate": sum(1 for m in all_metrics if m["resolved"]) / len(tasks) if tasks else 0,
        "per_task": all_metrics,
    }
    metrics_path.write_text(json.dumps(summary, indent=2))
    logger.info(f"Metrics saved to {metrics_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
