"""Task definitions — loaded from tasks.json at runtime."""

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Task:
    task_id: str
    repo_url: str
    commit: str
    problem_statement: str
    test_cmd: str
    test_patch: str | None = None

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        return cls(
            task_id=d["task_id"],
            repo_url=d["repo_url"],
            commit=d["commit"],
            problem_statement=d["problem_statement"],
            test_cmd=d["test_cmd"],
            test_patch=d.get("test_patch"),
        )


def get_tasks(path: str | Path = "tasks.json") -> list[Task]:
    with open(path) as f:
        data = json.load(f)
    return [Task.from_dict(t) for t in data["tasks"]]


def get_task_by_id(task_id: str, path: str | Path = "tasks.json") -> Task | None:
    for t in get_tasks(path):
        if t.task_id == task_id:
            return t
    return None
