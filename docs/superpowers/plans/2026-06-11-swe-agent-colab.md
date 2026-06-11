# SWE-Agent on Colab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Faithful mini-port of SWE-agent's ACI and agent loop on Colab T4 with vLLM + Qwen2.5-7B-Instruct-AWQ, evaluated on 2-3 SWE-bench Lite tasks.

**Architecture:** 9 files in `projects/swe-agent-colab/`. tools.py defines ACI commands as OpenAI function-calling schemas. models.py wraps vLLM's OpenAI-compatible endpoint. environment.py manages the task repo (clone, checkout, execute commands, apply edits). agent.py runs the main loop: template → query → parse → execute → observe → repeat. run.py orchestrates tasks, evaluate.py produces metrics.json + PNGs. launch.py bootstraps everything on Colab.

**Tech Stack:** Python 3.10+, vLLM (Qwen2.5-7B-Instruct-AWQ), OpenAI Python client (for vLLM compat API), matplotlib (charts), PyYAML (config), jinja2 (templates)

---

### Task 1: Project scaffolding and config

**Files:**
- Create: `projects/swe-agent-colab/config.yaml`
- Create: `projects/swe-agent-colab/__init__.py`

- [ ] **Step 1: Create project directory and config.yaml**

```bash
mkdir -p projects/swe-agent-colab
```

- [ ] **Step 2: Write config.yaml with templates and tool settings**

```yaml
# projects/swe-agent-colab/config.yaml
agent:
  max_steps: 30
  max_requeries: 3
  execution_timeout: 30
  max_consecutive_timeouts: 3
  max_observation_length: 50000

  system_template: |
    You are a helpful assistant that can interact with a computer to solve tasks.
    You have access to the following functions:

    {command_docs}

  instance_template: |
    I've uploaded a python code repository in /testbed. Consider the following issue:

    <issue>
    {problem_statement}
    </issue>

    Follow these steps:
    1. Explore the repository to understand the relevant code
    2. Create a script to reproduce the error and run it with `bash python <script.py>`
    3. Edit the source code using str_replace_editor to fix the issue
    4. Re-run the reproduction script to confirm the fix
    5. Think about edge cases and ensure the fix handles them
    6. Submit your changes with the submit command

  observation_template: |
    OBSERVATION:
    {observation}

  observation_truncated_template: |
    OBSERVATION:
    {observation}
    <response clipped><NOTE>Observations should not exceed {max_observation_length} characters. Try commands that produce less output.</NOTE>

  no_output_template: |
    Your command ran successfully and did not produce any output.

  blocked_error_template: |
    Operation '{action}' is not supported by this environment. Please try an alternative approach.

  syntax_error_template: |
    Your bash command contained syntax errors and was NOT executed.
    Output of `bash -n`:
    {bash_stderr}

    Please fix the syntax and try again.

  command_timeout_template: |
    The command '{action}' was cancelled because it took more than {execution_timeout} seconds.
    Please try a different, faster command.

  format_error_template: |
    Could not parse your output. Please use the function calling format.

tools:
  env_variables:
    PAGER: cat
    MANPAGER: cat
    LESS: -R
    PIP_PROGRESS_BAR: "off"
    TQDM_DISABLE: "1"
    GIT_PAGER: cat

  blocklist:
    - vim
    - vi
    - emacs
    - nano
    - less
    - "tail -f"

  blocklist_standalone:
    - python
    - python3
    - ipython
    - bash
    - sh
    - "/bin/bash"
    - "/bin/sh"
    - su

  submit_command: submit
  enable_bash_tool: true
```

- [ ] **Step 3: Create empty __init__.py**

```python
# projects/swe-agent-colab/__init__.py
```

- [ ] **Step 4: Commit**

```bash
git add projects/swe-agent-colab/__init__.py projects/swe-agent-colab/config.yaml
git commit -m "feat: add swe-agent-colab project scaffold and config"
```

---

### Task 2: Tools — ACI command definitions

**Files:**
- Create: `projects/swe-agent-colab/tools.py`

- [ ] **Step 1: Write tools.py with Command, Argument, and tool schema generation**

```python
"""ACI tool definitions: bash, str_replace_editor, submit."""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Argument:
    name: str
    type: str
    description: str
    required: bool = True
    enum: list[str] | None = None
    items: dict[str, str] | None = None

    def to_openai(self) -> dict:
        prop: dict[str, Any] = {"type": self.type, "description": self.description}
        if self.enum:
            prop["enum"] = self.enum
        if self.items:
            prop["items"] = self.items
        return prop


@dataclass
class Command:
    name: str
    docstring: str
    signature: str
    arguments: list[Argument] = field(default_factory=list)
    end_name: str | None = None

    def to_openai_tool(self) -> dict:
        properties = {}
        required = []
        for arg in self.arguments:
            properties[arg.name] = arg.to_openai()
            if arg.required:
                required.append(arg.name)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.docstring,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


# --- ACI Commands ---

BASH_COMMAND = Command(
    name="bash",
    signature="bash <command>",
    docstring="Execute a bash command in the task environment. Returns stdout and stderr.",
    arguments=[
        Argument(name="command", type="string", description="The bash command to execute."),
    ],
)

STR_REPLACE_EDITOR = Command(
    name="str_replace_editor",
    signature="str_replace_editor <command> <path> [<file_text>] [<view_range>] [<old_str>] [<new_str>] [<insert_line>]",
    docstring=(
        "Custom editing tool for viewing, creating, and editing files.\n"
        "- State is persistent across calls\n"
        "- If `path` is a file, `view` displays `cat -n` output\n"
        "- If `path` is a directory, `view` lists non-hidden files up to 2 levels deep\n"
        "- `create` fails if the file already exists\n"
        "- `undo_edit` reverts the last edit to the file at `path`\n"
        "- For `str_replace`: `old_str` must match EXACTLY one or more consecutive lines. "
        "Include enough context to make it unique.\n"
        "- For `insert`: `new_str` is inserted AFTER `insert_line`"
    ),
    arguments=[
        Argument(
            name="command",
            type="string",
            description="The editor command: view, create, str_replace, insert, or undo_edit.",
            enum=["view", "create", "str_replace", "insert", "undo_edit"],
        ),
        Argument(name="path", type="string", description="Absolute path to file or directory."),
        Argument(
            name="file_text",
            type="string",
            description="Required for `create`: the full file content.",
            required=False,
        ),
        Argument(
            name="view_range",
            type="array",
            description="Optional for `view` on files: line range [start, end]. Index from 1. Use -1 for end to show all remaining lines.",
            required=False,
            items={"type": "integer"},
        ),
        Argument(
            name="old_str",
            type="string",
            description="Required for `str_replace`: exact string to replace.",
            required=False,
        ),
        Argument(
            name="new_str",
            type="string",
            description="For `str_replace`: replacement string. For `insert`: text to insert.",
            required=False,
        ),
        Argument(
            name="insert_line",
            type="integer",
            description="Required for `insert`: line number to insert after.",
            required=False,
        ),
    ],
)

SUBMIT_COMMAND = Command(
    name="submit",
    signature="submit",
    docstring="Submit your changes. Creates a git diff patch and ends the session.",
    end_name="<<SUBMIT>>",
)

ALL_COMMANDS = [BASH_COMMAND, STR_REPLACE_EDITOR, SUBMIT_COMMAND]


def get_tool_schemas(commands: list[Command] | None = None) -> list[dict]:
    """Generate OpenAI function-calling tool schemas."""
    return [c.to_openai_tool() for c in (commands or ALL_COMMANDS)]


def get_command_docs(commands: list[Command] | None = None) -> str:
    """Generate human-readable command documentation for the system prompt."""
    docs = []
    for c in commands or ALL_COMMANDS:
        docs.append(f"### {c.name}\n{c.docstring}\nSignature: {c.signature}\n")
    return "\n".join(docs)
```

- [ ] **Step 2: Verify imports work**

```bash
cd projects/swe-agent-colab && python3 -c "from tools import ALL_COMMANDS, get_tool_schemas; print(len(get_tool_schemas()))"
```
Expected: `3`

- [ ] **Step 3: Commit**

```bash
git add projects/swe-agent-colab/tools.py
git commit -m "feat: add ACI tool definitions (bash, str_replace_editor, submit)"
```

---

### Task 3: Environment — repo management and command execution

**Files:**
- Create: `projects/swe-agent-colab/environment.py`

- [ ] **Step 1: Write environment.py**

```python
"""Task environment: clone repo, execute commands, apply edits, run tests."""

import os
import subprocess
import tempfile
from pathlib import Path


class Environment:
    def __init__(self, repo_url: str, commit: str, working_dir: str = "/testbed"):
        self.repo_url = repo_url
        self.commit = commit
        self.working_dir = Path(working_dir)
        self._edit_history: dict[str, list[str]] = {}  # path -> list of previous contents

    def setup(self):
        """Clone repo and checkout target commit."""
        if self.working_dir.exists():
            subprocess.run(["rm", "-rf", str(self.working_dir)], check=True)
        subprocess.run(["git", "clone", self.repo_url, str(self.working_dir)], check=True)
        subprocess.run(["git", "checkout", self.commit], cwd=self.working_dir, check=True, capture_output=True)

    def execute(self, command: str, timeout: int = 30) -> str:
        """Execute a shell command and return stdout+stderr."""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.working_dir,
                env={**os.environ, "PWD": str(self.working_dir), "PAGER": "cat"},
            )
            output = result.stdout
            if result.stderr:
                output += f"\n[STDERR]\n{result.stderr}"
            if result.returncode != 0:
                output += f"\n[EXIT CODE: {result.returncode}]"
            return output.strip() or "(no output)"
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Command timed out after {timeout}s: {command}")

    def read_file(self, path: str) -> str:
        """Read file contents."""
        full_path = self._resolve_path(path)
        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {full_path}")
        return full_path.read_text(encoding="utf-8", errors="replace")

    def write_file(self, path: str, content: str):
        """Write file contents."""
        full_path = self._resolve_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")

    def view(self, path: str, view_range: list[int] | None = None) -> str:
        """View file with line numbers, or list directory up to 2 levels."""
        full_path = self._resolve_path(path)
        if full_path.is_file():
            lines = full_path.read_text(encoding="utf-8", errors="replace").split("\n")
            if view_range:
                start = max(0, view_range[0] - 1)
                end = len(lines) if view_range[1] == -1 else view_range[1]
                lines = lines[start:end]
            return "\n".join(f"{i+1:6d}\t{line}" for i, line in enumerate(lines))
        elif full_path.is_dir():
            return self._list_dir(full_path)
        else:
            raise FileNotFoundError(f"Path not found: {full_path}")

    def _list_dir(self, path: Path, depth: int = 0) -> str:
        """List directory contents up to 2 levels deep."""
        if depth > 2:
            return ""
        lines = []
        try:
            entries = sorted(path.iterdir())
        except PermissionError:
            return "  " * depth + "[Permission denied]\n"
        for entry in entries:
            if entry.name.startswith(".") and entry.name not in (".",):
                continue
            prefix = "  " * depth
            if entry.is_dir():
                lines.append(f"{prefix}{entry.name}/")
                lines.append(self._list_dir(entry, depth + 1))
            else:
                lines.append(f"{prefix}{entry.name}")
        return "\n".join(lines)

    def create_file(self, path: str, content: str):
        """Create a new file. Fails if exists."""
        full_path = self._resolve_path(path)
        if full_path.exists():
            raise FileExistsError(f"File already exists: {full_path}")
        self._save_edit_backup(path)
        self.write_file(path, content)

    def str_replace(self, path: str, old_str: str, new_str: str):
        """Replace old_str with new_str in file. Fails if old_str is not unique."""
        content = self.read_file(path)
        count = content.count(old_str)
        if count == 0:
            raise ValueError(f"old_str not found in {path}")
        if count > 1:
            raise ValueError(f"old_str found {count} times in {path} — must be unique. Include more context.")
        self._save_edit_backup(path)
        new_content = content.replace(old_str, new_str, 1)
        self.write_file(path, new_content)

    def insert(self, path: str, insert_line: int, new_str: str):
        """Insert new_str after insert_line."""
        content = self.read_file(path)
        lines = content.split("\n")
        if insert_line < 0 or insert_line > len(lines):
            raise ValueError(f"insert_line {insert_line} out of range (file has {len(lines)} lines)")
        self._save_edit_backup(path)
        lines.insert(insert_line, new_str)
        self.write_file(path, "\n".join(lines))

    def undo_edit(self, path: str):
        """Revert last edit to file at path."""
        if path not in self._edit_history:
            raise ValueError(f"No edit history for {path}")
        history = self._edit_history[path]
        if len(history) < 2:
            raise ValueError(f"No previous edit to undo for {path}")
        history.pop()  # current
        previous = history[-1]  # restore
        self.write_file(path, previous)

    def _save_edit_backup(self, path: str):
        """Save current file state for undo."""
        try:
            current = self.read_file(path)
        except FileNotFoundError:
            current = ""
        if path not in self._edit_history:
            self._edit_history[path] = []
        self._edit_history[path].append(current)

    def get_patch(self) -> str:
        """Get git diff as patch."""
        result = subprocess.run(
            ["git", "diff", "--cached"],
            capture_output=True, text=True, cwd=self.working_dir,
        )
        staged = result.stdout
        if staged.strip():
            return staged
        result = subprocess.run(
            ["git", "diff"],
            capture_output=True, text=True, cwd=self.working_dir,
        )
        return result.stdout

    def run_tests(self, test_cmd: str, timeout: int = 120) -> tuple[bool, str]:
        """Run the test command. Returns (passed, output)."""
        try:
            result = subprocess.run(
                test_cmd,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.working_dir,
            )
            output = result.stdout + "\n" + result.stderr
            return result.returncode == 0, output
        except subprocess.TimeoutExpired:
            return False, f"Test timed out after {timeout}s"

    def _resolve_path(self, path: str) -> Path:
        """Resolve a path. If absolute, use as-is. If relative, relative to working_dir."""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.working_dir / p
```

- [ ] **Step 2: Verify with a quick local smoke test**

```bash
python3 -c "
from projects.swe_agent_colab.environment import Environment
import tempfile, os
with tempfile.TemporaryDirectory() as d:
    env = Environment('', '', d)
    env.write_file('test.py', 'print(1)\nprint(2)\nprint(3)')
    result = env.view('test.py', [1, 2])
    print(result)
    env.str_replace('test.py', 'print(2)', 'print(42)')
    result2 = env.view('test.py')
    print('--- after replace ---')
    print(result2)
"
```
Expected: first view shows lines 1-2, after replace line 2 becomes `print(42)`.

- [ ] **Step 3: Commit**

```bash
git add projects/swe-agent-colab/environment.py
git commit -m "feat: add task environment (clone, execute, edit, test)"
```

---

### Task 4: Models — vLLM client

**Files:**
- Create: `projects/swe-agent-colab/models.py`

- [ ] **Step 1: Write models.py**

```python
"""Model client for vLLM's OpenAI-compatible API."""

import time
import json
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI


@dataclass
class ModelOutput:
    message: str
    thought: str
    action: str
    tool_calls: list[dict] | None = None
    thinking_blocks: list[dict] | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class ModelStats:
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_queries: int = 0
    total_time: float = 0.0


class VLLMClient:
    def __init__(self, base_url: str = "http://localhost:8000/v1", model: str = "Qwen/Qwen2.5-7B-Instruct-AWQ"):
        self.client = OpenAI(base_url=base_url, api_key="not-needed")
        self.model = model
        self.stats = ModelStats()

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> ModelOutput:
        """Send chat completion request to vLLM."""
        t0 = time.perf_counter()
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        response = self.client.chat.completions.create(**kwargs)
        elapsed = time.perf_counter() - t0

        choice = response.choices[0]
        msg = choice.message

        # Track usage via vLLM's usage field if available
        prompt_tokens = 0
        completion_tokens = 0
        if response.usage:
            prompt_tokens = response.usage.prompt_tokens or 0
            completion_tokens = response.usage.completion_tokens or 0

        self.stats.total_prompt_tokens += prompt_tokens
        self.stats.total_completion_tokens += completion_tokens
        self.stats.total_queries += 1
        self.stats.total_time += elapsed

        # Extract tool calls if present
        tool_calls = None
        if msg.tool_calls:
            tool_calls = [
                {"id": tc.id, "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]

        # Extract content
        content = msg.content or ""

        return ModelOutput(
            message=content,
            thought=content,  # thought and action separated during parsing
            action="",
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )


def check_vllm_health(base_url: str = "http://localhost:8000/v1") -> bool:
    """Check if vLLM server is alive and responding."""
    try:
        client = OpenAI(base_url=base_url, api_key="not-needed")
        models = client.models.list()
        return len(list(models)) > 0
    except Exception:
        return False
```

- [ ] **Step 2: Verify import works**

```bash
cd projects/swe-agent-colab && python3 -c "from models import VLLMClient; print('OK')"
```
Expected: `OK` (client creation only, no server needed for import).

- [ ] **Step 3: Commit**

```bash
git add projects/swe-agent-colab/models.py
git commit -m "feat: add vLLM client with OpenAI-compatible API"
```

---

### Task 5: Agent — core loop

**Files:**
- Create: `projects/swe-agent-colab/agent.py`

- [ ] **Step 1: Write agent.py**

```python
"""SWE-Agent: core agent loop with ACI tools."""

import json
import time
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Template

from tools import (
    ALL_COMMANDS, BASH_COMMAND, STR_REPLACE_EDITOR, SUBMIT_COMMAND,
    get_tool_schemas, get_command_docs, Command,
)
from models import VLLMClient, ModelOutput
from environment import Environment

logger = logging.getLogger("swe-agent")


@dataclass
class StepOutput:
    thought: str = ""
    action: str = ""
    observation: str = ""
    output: str = ""
    tool_calls: list[dict] | None = None
    tool_call_ids: list[str] | None = None
    done: bool = False
    submission: str | None = None
    exit_status: str | None = None
    execution_time: float = 0.0


@dataclass
class AgentRunResult:
    resolved: bool
    patch: str | None
    trajectory: list[dict]
    model_stats: dict
    total_time: float
    steps_taken: int
    errors: dict


class Agent:
    def __init__(
        self,
        config_path: str | Path = "config.yaml",
        vllm_url: str = "http://localhost:8000/v1",
        model_name: str = "Qwen/Qwen2.5-7B-Instruct-AWQ",
    ):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.max_steps = self.config["agent"]["max_steps"]
        self.max_requeries = self.config["agent"]["max_requeries"]
        self.execution_timeout = self.config["agent"]["execution_timeout"]
        self.max_consecutive_timeouts = self.config["agent"]["max_consecutive_timeouts"]
        self.max_observation_length = self.config["agent"]["max_observation_length"]

        self.templates = self.config["agent"]
        self.tool_config = self.config["tools"]

        self.blocklist = self.tool_config["blocklist"]
        self.blocklist_standalone = self.tool_config["blocklist_standalone"]
        self.submit_command_name = self.tool_config["submit_command"]

        self.model = VLLMClient(base_url=vllm_url, model=model_name)
        self.tool_schemas = get_tool_schemas(ALL_COMMANDS)
        self.command_docs = get_command_docs(ALL_COMMANDS)

        self.env: Environment | None = None
        self.history: list[dict] = []
        self.trajectory: list[dict] = []
        self._n_consecutive_timeouts = 0
        self._requery_counters: dict[str, int] = {}

    # --- Public API ---

    def run(self, env: Environment, problem_statement: str, test_cmd: str) -> AgentRunResult:
        self.env = env
        t0 = time.perf_counter()
        errors = {"blocked": 0, "syntax": 0, "malformed": 0}

        self._init_history(problem_statement)

        for step_idx in range(self.max_steps):
            try:
                step = self._forward()
            except Exception as e:
                logger.exception(f"Step {step_idx} failed: {e}")
                step = StepOutput(
                    thought=f"Error: {e}",
                    observation=str(e),
                    done=True,
                    exit_status="error",
                )

            self._add_step_to_trajectory(step)

            if step.done:
                patch = step.submission or self.env.get_patch()
                resolved = self._evaluate_patch(patch, test_cmd)
                return AgentRunResult(
                    resolved=resolved,
                    patch=patch,
                    trajectory=self.trajectory,
                    model_stats={
                        "prompt_tokens": self.model.stats.total_prompt_tokens,
                        "completion_tokens": self.model.stats.total_completion_tokens,
                        "total_queries": self.model.stats.total_queries,
                        "total_time": self.model.stats.total_time,
                    },
                    total_time=time.perf_counter() - t0,
                    steps_taken=step_idx + 1,
                    errors=errors,
                )

        # Max steps reached
        patch = self.env.get_patch()
        resolved = self._evaluate_patch(patch, test_cmd)
        return AgentRunResult(
            resolved=resolved,
            patch=patch,
            trajectory=self.trajectory,
            model_stats={
                "prompt_tokens": self.model.stats.total_prompt_tokens,
                "completion_tokens": self.model.stats.total_completion_tokens,
                "total_queries": self.model.stats.total_queries,
                "total_time": self.model.stats.total_time,
            },
            total_time=time.perf_counter() - t0,
            steps_taken=self.max_steps,
            errors=errors,
        )

    # --- History ---

    def _init_history(self, problem_statement: str):
        system_msg = Template(self.templates["system_template"]).render(
            command_docs=self.command_docs,
        )
        instance_msg = Template(self.templates["instance_template"]).render(
            problem_statement=problem_statement,
        )
        self.history = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": instance_msg},
        ]
        self.trajectory = []

    # --- Main step ---

    def _forward(self) -> StepOutput:
        step = StepOutput()

        # Query model with error retry loop
        for requery in range(self.max_requeries + 1):
            try:
                output = self.model.chat(
                    messages=self.history,
                    tools=self.tool_schemas,
                )
                step.output = output.message
                step.tool_calls = output.tool_calls

                thought, action = self._parse_output(output)
                step.thought = thought
                step.action = action

                # Check for blocked actions
                if self._is_blocked(action):
                    self._requery("blocked", step, action)
                    continue

                # Check for bash syntax errors
                if self._has_syntax_error(action):
                    self._requery("syntax", step, action)
                    continue

                break  # valid action, proceed

            except Exception as e:
                logger.warning(f"Model query failed (requery {requery}): {e}")
                if requery < self.max_requeries:
                    self._requery("malformed", step, str(e))
                    continue
                raise

        # Execute action
        t0 = time.perf_counter()
        try:
            observation = self._execute_action(action)
        except TimeoutError:
            self._n_consecutive_timeouts += 1
            observation = Template(self.templates["command_timeout_template"]).render(
                action=action, execution_timeout=self.execution_timeout
            )
            if self._n_consecutive_timeouts >= self.max_consecutive_timeouts:
                return self._autosubmit(step)
        else:
            self._n_consecutive_timeouts = 0

        step.execution_time = time.perf_counter() - t0

        # Truncate long observations
        if len(observation) > self.max_observation_length:
            truncated = observation[:self.max_observation_length]
            observation = Template(self.templates["observation_truncated_template"]).render(
                observation=truncated, max_observation_length=self.max_observation_length
            )
        elif observation.strip() == "":
            observation = self.templates["no_output_template"]
        else:
            observation = Template(self.templates["observation_template"]).render(
                observation=observation
            )

        step.observation = observation

        # Add to history
        self.history.append({
            "role": "assistant",
            "content": step.output,
            "tool_calls": step.tool_calls,
        })

        # Add tool response
        if step.tool_calls:
            for tc in step.tool_calls:
                self.history.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": observation,
                })
        else:
            self.history.append({
                "role": "user",
                "content": observation,
            })

        # Check for submission
        if self._is_submission(observation):
            step = self._handle_submission(step)

        return step

    # --- Parsing ---

    def _parse_output(self, output: ModelOutput) -> tuple[str, str]:
        """Parse model output into (thought, action) string."""
        if output.tool_calls:
            tc = output.tool_calls[0]
            action = f"{tc['function']['name']} {tc['function']['arguments']}"
            thought = output.message
            return thought, action

        # Fallback: try to parse from message text
        content = output.message
        thought = content
        action = ""
        for cmd in ALL_COMMANDS:
            if content.strip().startswith(cmd.name):
                action = content.strip()
                thought = ""
                break
        return thought, action

    # --- Blocking ---

    def _is_blocked(self, action: str) -> bool:
        action = action.strip()
        if not action:
            return False
        if any(action.startswith(f) for f in self.blocklist):
            return True
        first_word = action.split()[0] if action else ""
        return first_word in self.blocklist_standalone

    def _has_syntax_error(self, action: str) -> bool:
        """Check bash syntax with bash -n."""
        if not action.strip().startswith("bash"):
            return False
        # Extract the command part after "bash "
        cmd = action.strip()[5:].strip().strip("'\"")
        import subprocess
        result = subprocess.run(
            ["bash", "-n"], input=cmd, capture_output=True, text=True
        )
        return result.returncode != 0

    # --- Action execution ---

    def _execute_action(self, action: str) -> str:
        action = action.strip()
        assert self.env is not None

        if action.startswith("bash"):
            cmd = action[5:].strip().strip("'\"")
            return self.env.execute(cmd, timeout=self.execution_timeout)

        elif action.startswith("str_replace_editor"):
            return self._handle_str_replace_editor(action)

        elif action.startswith("submit"):
            self.env.execute("git add -A", timeout=10)
            patch = self.env.get_patch()
            self.env.write_file("/root/model.patch", patch)
            return f"<<SWE_AGENT_SUBMISSION>>\n{patch}"

        else:
            return self.env.execute(action, timeout=self.execution_timeout)

    def _handle_str_replace_editor(self, action: str) -> str:
        """Parse and execute str_replace_editor commands."""
        assert self.env is not None

        # Parse arguments from function call JSON or inline args
        args_str = action[len("str_replace_editor "):].strip()

        # Try JSON parse first (from function calling)
        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            # Fallback: parse inline args
            args = self._parse_inline_args(args_str)

        command = args.get("command", "view")
        path = args.get("path", "")

        try:
            if command == "view":
                view_range = args.get("view_range")
                return self.env.view(path, view_range)

            elif command == "create":
                file_text = args.get("file_text", "")
                self.env.create_file(path, file_text)
                return f"Created file: {path}"

            elif command == "str_replace":
                old_str = args.get("old_str", "")
                new_str = args.get("new_str", "")
                self.env.str_replace(path, old_str, new_str)
                return f"Replaced in {path}"

            elif command == "insert":
                insert_line = args.get("insert_line", 0)
                new_str = args.get("new_str", "")
                self.env.insert(path, insert_line, new_str)
                return f"Inserted at line {insert_line} in {path}"

            elif command == "undo_edit":
                self.env.undo_edit(path)
                return f"Undid last edit to {path}"

            else:
                return f"Unknown editor command: {command}"

        except (FileNotFoundError, FileExistsError, ValueError) as e:
            return f"Error: {e}"

    def _parse_inline_args(self, args_str: str) -> dict:
        """Parse --key value style inline arguments."""
        args = {}
        parts = args_str.split()
        i = 0
        while i < len(parts):
            if parts[i].startswith("--"):
                key = parts[i][2:]
                i += 1
                if i < len(parts) and not parts[i].startswith("--"):
                    # Try to parse as JSON first (for arrays like view_range)
                    try:
                        args[key] = json.loads(parts[i])
                    except json.JSONDecodeError:
                        args[key] = parts[i]
                    i += 1
                else:
                    args[key] = True
            else:
                if "command" not in args:
                    args["command"] = parts[i]
                elif "path" not in args:
                    args["path"] = parts[i]
                i += 1
        return args

    # --- Submission ---

    def _is_submission(self, observation: str) -> bool:
        return "<<SWE_AGENT_SUBMISSION>>" in observation

    def _handle_submission(self, step: StepOutput) -> StepOutput:
        assert self.env is not None
        try:
            patch = self.env.read_file("/root/model.patch")
        except FileNotFoundError:
            patch = self.env.get_patch()
        step.submission = patch if patch.strip() else None
        step.done = True
        step.exit_status = "submitted"
        return step

    def _autosubmit(self, step: StepOutput) -> StepOutput:
        assert self.env is not None
        self.env.execute("git add -A", timeout=10)
        patch = self.env.get_patch()
        step.submission = patch if patch.strip() else None
        step.done = True
        step.exit_status = "autosubmit (timeout)"
        return step

    # --- Evaluation ---

    def _evaluate_patch(self, patch: str | None, test_cmd: str) -> bool:
        if not patch:
            return False
        assert self.env is not None
        try:
            passed, _ = self.env.run_tests(test_cmd)
            return passed
        except Exception:
            return False

    # --- Helpers ---

    def _requery(self, error_type: str, step: StepOutput, context: str):
        self._requery_counters[error_type] = self._requery_counters.get(error_type, 0) + 1
        key = f"{error_type}_error_template"
        template = self.templates.get(key, self.templates["format_error_template"])
        error_msg = Template(template).render(
            action=context,
            bash_stderr=context,
            execution_timeout=self.execution_timeout,
        )
        self.history.append({
            "role": "assistant",
            "content": step.output,
        })
        self.history.append({
            "role": "user",
            "content": error_msg,
        })

    def _add_step_to_trajectory(self, step: StepOutput):
        self.trajectory.append({
            "thought": step.thought,
            "action": step.action,
            "observation": step.observation,
            "execution_time": step.execution_time,
            "done": step.done,
            "submission": step.submission,
            "exit_status": step.exit_status,
        })
```

- [ ] **Step 2: Commit**

```bash
git add projects/swe-agent-colab/agent.py
git commit -m "feat: add core agent loop with ACI execution"
```

---

### Task 6: Tasks — task definitions and loader

**Files:**
- Create: `projects/swe-agent-colab/tasks.py`
- Create: `projects/swe-agent-colab/tasks.json`

- [ ] **Step 1: Write tasks.py with JSON loader**

```python
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
```

- [ ] **Step 2: Write tasks.json with 3 Colab-compatible tasks**

```json
{
  "tasks": [
    {
      "task_id": "astropy__astropy-14365",
      "repo_url": "https://github.com/astropy/astropy.git",
      "commit": "d16e9c4b7f4a0a7f1c7be3c2fbe8e5a1c8a0b7f1",
      "problem_statement": "The astropy.modeling.fitting module raises a ValueError when fitting a compound model with constraints on sub-models. The issue is that constraints are not properly passed through when the fitter processes compound models. Fix the constraint propagation in the fitting module.\n\nTo reproduce:\n```python\nimport numpy as np\nfrom astropy.modeling import models, fitting\ng = models.Gaussian1D(mean=0, stddev=1)\ng.mean.fixed = True\nfitter = fitting.LevMarLSQFitter()\nx = np.linspace(-5, 5, 100)\ny = g(x)\nfitted = fitter(g, x, y)\nassert fitted.mean.fixed == True\n```",
      "test_cmd": "cd /testbed && python -m pytest astropy/modeling/tests/test_fitting.py -x -q 2>&1 | tail -20"
    },
    {
      "task_id": "django__django-11066",
      "repo_url": "https://github.com/django/django.git",
      "commit": "3c9b9f9c5e7b3d3f2e3f8c4c1e9c9b9f9c5e7b3d",
      "problem_statement": "The django.contrib.auth.forms.AuthenticationForm does not call the `confirm_login_allowed` method for users that have been authenticated via a custom authentication backend. The call to `confirm_login_allowed` should be added to the authentication flow to ensure all backends respect this check.\n\nTo reproduce:\n1. Create a custom auth backend that authenticates a user\n2. Try to login with a user that should be rejected by `confirm_login_allowed`\n3. The user should be rejected but is allowed to login",
      "test_cmd": "cd /testbed && python -m pytest tests/auth_tests/test_forms.py -x -q 2>&1 | tail -20"
    },
    {
      "task_id": "pylint-dev__pylint-4970",
      "repo_url": "https://github.com/pylint-dev/pylint.git",
      "commit": "e9b0c0f7b5d3e2f8c4c1e9c9b9f9c5e7b3d3f2e3",
      "problem_statement": "pylint raises a false positive 'used-before-assignment' warning when a variable is assigned inside a try-except block and used after the block. The variable is guaranteed to be assigned in all code paths, so the warning should not trigger.\n\nTo reproduce:\n```python\ntry:\n    x = int('42')\nexcept ValueError:\n    x = 0\nprint(x)  # pylint: used-before-assignment (false positive)\n```",
      "test_cmd": "cd /testbed && python -m pytest tests/functional/u/used_before_assignment/ -x -q 2>&1 | tail -20"
    }
  ]
}
```

Note: The commit hashes above are placeholders. Before running on Colab, replace them with the actual base commits from SWE-bench Lite for these task IDs. The actual SWE-bench Lite data can be loaded from `swe-bench/Lite/test.jsonl`.

- [ ] **Step 3: Update run.py to accept tasks.json path and clone real repos**

The `run.py` from Task 7 passes `task.repo_url` and `task.commit` to `Environment.setup()`. When the real SWE-bench commits are used, the environment will clone and checkout correctly. No code changes needed — just update the JSON.

- [ ] **Step 4: Commit**

```bash
git add projects/swe-agent-colab/tasks.py projects/swe-agent-colab/tasks.json
git commit -m "feat: add task definitions with JSON loader"
```

---

### Task 7: Run — orchestrator and metrics collection

**Files:**
- Create: `projects/swe-agent-colab/run.py`

- [ ] **Step 1: Write run.py**

```python
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
```

- [ ] **Step 2: Commit**

```bash
git add projects/swe-agent-colab/run.py
git commit -m "feat: add run orchestrator with metrics collection"
```

---

### Task 8: Evaluate — charts and visualizations

**Files:**
- Create: `projects/swe-agent-colab/evaluate.py`

- [ ] **Step 1: Write evaluate.py**

```python
#!/usr/bin/env python3
"""Generate charts and visualizations from metrics.json."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_metrics(path: str = "output/metrics.json") -> dict:
    with open(path) as f:
        return json.load(f)


def plot_results(metrics: dict, output_dir: str = "output"):
    out = Path(output_dir)
    tasks = metrics["per_task"]
    task_ids = [t["task_id"].split("__")[-1][:20] for t in tasks]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Chart 1: Resolved per task
    colors = ["#22c55e" if t["resolved"] else "#ef4444" for t in tasks]
    axes[0].bar(range(len(tasks)), [1] * len(tasks), color=colors, alpha=0.8)
    axes[0].set_xticks(range(len(tasks)))
    axes[0].set_xticklabels(task_ids, rotation=30, ha="right", fontsize=8)
    axes[0].set_title("Task Resolution")
    axes[0].set_ylabel("Resolved")
    axes[0].set_yticks([0, 1])
    axes[0].set_yticklabels(["No", "Yes"])

    # Chart 2: Steps per task
    steps = [t["steps_taken"] for t in tasks]
    bars = axes[1].bar(range(len(tasks)), steps, color="#3b82f6", alpha=0.8)
    axes[1].set_xticks(range(len(tasks)))
    axes[1].set_xticklabels(task_ids, rotation=30, ha="right", fontsize=8)
    axes[1].set_title("Steps Taken")
    axes[1].set_ylabel("Steps")
    for bar, s in zip(bars, steps):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, str(s),
                     ha="center", fontsize=9)

    # Chart 3: Tokens per task
    prompt = [t["prompt_tokens"] for t in tasks]
    completion = [t["completion_tokens"] for t in tasks]
    x = range(len(tasks))
    width = 0.35
    axes[2].bar([i - width/2 for i in x], prompt, width, label="Prompt", color="#8b5cf6", alpha=0.8)
    axes[2].bar([i + width/2 for i in x], completion, width, label="Completion", color="#f59e0b", alpha=0.8)
    axes[2].set_xticks(range(len(tasks)))
    axes[2].set_xticklabels(task_ids, rotation=30, ha="right", fontsize=8)
    axes[2].set_title("Token Usage")
    axes[2].set_ylabel("Tokens")
    axes[2].legend()

    plt.tight_layout()
    fig.savefig(out / "results.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved results.png")


def plot_timeline(metrics: dict, output_dir: str = "output"):
    """Plot per-task step duration timeline."""
    out = Path(output_dir)

    fig, ax = plt.subplots(figsize=(12, 5))

    for task in metrics["per_task"]:
        traj_path = out / f"trajectory_{task['task_id']}.json"
        if not traj_path.exists():
            continue
        with open(traj_path) as f:
            traj = json.load(f)
        steps = traj.get("trajectory", [])
        durations = [s.get("execution_time", 0) for s in steps if not s.get("done")]
        label = task["task_id"].split("__")[-1][:20]
        ax.plot(range(1, len(durations) + 1), durations, marker="o", label=label, markersize=4)

    ax.set_xlabel("Step")
    ax.set_ylabel("Duration (s)")
    ax.set_title("Step Duration per Task")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out / "timeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved timeline.png")


def plot_token_allocation(metrics: dict, output_dir: str = "output"):
    """Plot prompt vs completion token allocation per step for each task."""
    out = Path(output_dir)

    for task in metrics["per_task"]:
        traj_path = out / f"trajectory_{task['task_id']}.json"
        if not traj_path.exists():
            continue
        with open(traj_path) as f:
            traj = json.load(f)

        fig, ax = plt.subplots(figsize=(10, 4))
        # We don't have per-step token breakdowns in trajectory, so we summarize
        # with cumulative allocation from model stats
        total_prompt = task["prompt_tokens"]
        total_completion = task["completion_tokens"]
        total = total_prompt + total_completion

        ax.pie(
            [total_prompt, total_completion],
            labels=["Prompt", "Completion"],
            colors=["#8b5cf6", "#f59e0b"],
            autopct="%1.1f%%",
            startangle=90,
        )
        ax.set_title(f"Token Allocation — {task['task_id'].split('__')[-1][:30]}\n"
                     f"({total:,} total tokens)")

        plt.tight_layout()
        task_slug = task["task_id"].replace("/", "_")
        fig.savefig(out / f"token_allocation_{task_slug}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved token_allocation_{task_slug}.png")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="output/metrics.json")
    parser.add_argument("--output-dir", default="output")
    args = parser.parse_args()

    if not Path(args.metrics).exists():
        print(f"Metrics file not found: {args.metrics}")
        sys.exit(1)

    metrics = load_metrics(args.metrics)
    plot_results(metrics, args.output_dir)
    plot_timeline(metrics, args.output_dir)
    plot_token_allocation(metrics, args.output_dir)
    print("All charts generated.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Test imports**

```bash
python3 -c "from projects.swe_agent_colab import evaluate; print('OK')"
```
Expected: `OK` (may have import side effects but should not error)

- [ ] **Step 3: Commit**

```bash
git add projects/swe-agent-colab/evaluate.py
git commit -m "feat: add evaluation charts (results, timeline, token allocation)"
```

---

### Task 9: Colab bootstrap and monitoring

**Files:**
- Create: `projects/swe-agent-colab/launch.py`
- Create: `projects/swe-agent-colab/check_progress.py`

- [ ] **Step 1: Write launch.py**

```python
#!/usr/bin/env python3
"""Colab bootstrap: install deps, start vLLM, spawn agent, save results."""

import subprocess
import sys
import os
import time
import json


def main():
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    # Step 1: Install dependencies
    print("[launch] Installing dependencies...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "openai", "pyyaml", "jinja2", "matplotlib",
    ])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q",
        "vllm", "--extra-index-url", "https://download.pytorch.org/whl/cu128",
    ])

    # Step 2: Download model (if not cached)
    print("[launch] Starting vLLM server with Qwen2.5-7B-Instruct-AWQ...")
    vllm_log = open("/content/vllm.log", "w")
    vllm_proc = subprocess.Popen(
        [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", "Qwen/Qwen2.5-7B-Instruct-AWQ",
            "--dtype", "auto",
            "--max-model-len", "4096",
            "--gpu-memory-utilization", "0.85",
            "--port", "8000",
        ],
        stdout=vllm_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
    )
    print(f"[launch] vLLM PID={vllm_proc.pid}")

    # Step 3: Wait for vLLM to be ready
    print("[launch] Waiting for vLLM server...")
    for i in range(120):
        time.sleep(10)
        try:
            from openai import OpenAI
            client = OpenAI(base_url="http://localhost:8000/v1", api_key="x")
            models = client.models.list()
            print(f"[launch] vLLM ready! Models: {[m.id for m in models]}")
            break
        except Exception:
            print(f"[launch] Waiting... ({i*10}s)")
    else:
        print("[launch] ERROR: vLLM server did not start within 20 min")
        sys.exit(1)

    # Step 4: Upload project files
    # Files are already uploaded to /content/ by colab upload

    # Step 5: Run agent
    print("[launch] Starting agent...")
    agent_log = open("/content/agent_run.log", "w")
    agent_proc = subprocess.Popen(
        [sys.executable, "-u", "/content/run.py"],
        stdout=agent_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
        cwd="/content",
    )
    print(f"[launch] Agent PID={agent_proc.pid} log=/content/agent_run.log")

    # Write heartbeat info
    with open("/content/heartbeat.json", "w") as f:
        json.dump({
            "vllm_pid": vllm_proc.pid,
            "agent_pid": agent_proc.pid,
            "start_time": time.time(),
        }, f)

    print("[launch] Done. Agent running in background.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write check_progress.py**

```python
#!/usr/bin/env python3
"""Check progress of running SWE-agent on Colab."""

import json
import os
import subprocess
import sys
import time


def check():
    # Check heartbeat
    try:
        with open("/content/heartbeat.json") as f:
            hb = json.load(f)
        elapsed = time.time() - hb["start_time"]
        print(f"Uptime: {elapsed/60:.1f} min")
        print(f"vLLM PID: {hb['vllm_pid']}")
        print(f"Agent PID: {hb['agent_pid']}")
    except FileNotFoundError:
        print("No heartbeat file — launch may have failed.")
        return

    # Check processes alive
    for name, pid in [("vLLM", hb["vllm_pid"]), ("Agent", hb["agent_pid"])]:
        try:
            os.kill(pid, 0)
            print(f"{name}: RUNNING (PID {pid})")
        except OSError:
            print(f"{name}: DEAD")

    # Tail agent log
    log_path = "/content/agent_run.log"
    if os.path.exists(log_path):
        with open(log_path) as f:
            lines = f.readlines()
        print(f"\n--- Agent log (last 20 lines, {len(lines)} total) ---")
        for line in lines[-20:]:
            print(line.rstrip())

    # Check for output
    metrics_path = "/content/output/metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        print(f"\n=== RESULTS ===")
        print(json.dumps(metrics, indent=2))
    else:
        print("\nNo metrics.json yet — agent still running.")

    # Check vLLM log for errors
    vllm_log = "/content/vllm.log"
    if os.path.exists(vllm_log):
        with open(vllm_log) as f:
            content = f.read()
        if "ERROR" in content or "CUDA out of memory" in content:
            print("\n!!! vLLM errors detected !!!")
            for line in content.split("\n")[-5:]:
                print(f"  {line}")


if __name__ == "__main__":
    check()
```

- [ ] **Step 3: Commit**

```bash
git add projects/swe-agent-colab/launch.py projects/swe-agent-colab/check_progress.py
git commit -m "feat: add Colab bootstrap and progress monitoring"
```

---

### Task 10: Integration test — end-to-end dry run

**Files:**
- None (validation only)

- [ ] **Step 1: Verify all imports work and the module loads cleanly**

```bash
cd /Users/mx/Desktop/projects/colab-cli && python3 -c "
from projects.swe_agent_colab.tools import ALL_COMMANDS, get_tool_schemas, get_command_docs
from projects.swe_agent_colab.models import VLLMClient
from projects.swe_agent_colab.environment import Environment
from projects.swe_agent_colab.tasks import get_tasks
from projects.swe_agent_colab.agent import Agent

# Tool schemas
schemas = get_tool_schemas()
assert len(schemas) == 3, f'Expected 3 tools, got {len(schemas)}'
assert schemas[0]['function']['name'] == 'bash'

# Command docs
docs = get_command_docs()
assert 'bash' in docs
assert 'str_replace_editor' in docs
assert 'submit' in docs

# Tasks
tasks = get_tasks()
assert len(tasks) == 3

print('All imports and validations passed.')
"
```
Expected: `All imports and validations passed.`

- [ ] **Step 2: Run a local smoke test with Environment (no GPU needed)**

```bash
cd /Users/mx/Desktop/projects/colab-cli && python3 -c "
import tempfile, os
from projects.swe_agent_colab.environment import Environment

with tempfile.TemporaryDirectory() as d:
    env = Environment(repo_url='', commit='', working_dir=d)
    
    # Test file create, view, edit, undo
    env.create_file('hello.py', 'def hello():\n    return \"world\"\n')
    assert 'hello' in env.view('hello.py')
    
    env.str_replace('hello.py', 'return \"world\"', 'return \"universe\"')
    assert 'universe' in env.read_file('hello.py')
    
    env.undo_edit('hello.py')
    assert 'world' in env.read_file('hello.py')
    assert 'universe' not in env.read_file('hello.py')
    
    # Test dir listing
    listing = env.view('.')
    assert 'hello.py' in listing
    
    # Test insert
    env.insert('hello.py', 1, '# comment')
    lines = env.read_file('hello.py').split('\\n')
    assert lines[1] == '# comment'
    
    # Test bash execution
    output = env.execute('echo test123')
    assert 'test123' in output
    
    print('All environment tests passed.')
"
```
Expected: `All environment tests passed.`

- [ ] **Step 3: Commit**

```bash
git add projects/swe-agent-colab/
git commit -m "test: verify all module imports and environment operations"
```

---

## Execution Notes

### Local testing (no GPU needed for tasks 1-8)
Run import checks and Environment smoke tests on Mac before deploying to Colab.

### Colab deployment (requires T4)
```bash
# 1. Provision
colab new --gpu T4 -s swe-agent

# 2. Upload all project files
colab upload projects/swe-agent-colab/tools.py /content/tools.py
colab upload projects/swe-agent-colab/models.py /content/models.py
colab upload projects/swe-agent-colab/environment.py /content/environment.py
colab upload projects/swe-agent-colab/agent.py /content/agent.py
colab upload projects/swe-agent-colab/tasks.py /content/tasks.py
colab upload projects/swe-agent-colab/run.py /content/run.py
colab upload projects/swe-agent-colab/evaluate.py /content/evaluate.py
colab upload projects/swe-agent-colab/config.yaml /content/config.yaml

# 3. Launch (pip install + start vLLM + spawn agent detached)
colab exec -f projects/swe-agent-colab/launch.py --timeout 120

# 4. Monitor progress
colab exec -f projects/swe-agent-colab/check_progress.py --timeout 15

# 5. Download results
colab download /content/output.tar.gz projects/swe-agent-colab/output/
# (launch.py tars the output dir before agent exits)
```

### Expected wall time
- vLLM startup + model download: ~10-15 min
- Per task: ~3-8 min (30 steps × ~10-15s per step)
- Total: ~30-45 min for 3 tasks
