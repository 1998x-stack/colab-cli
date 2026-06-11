"""SWE-Agent: core agent loop with ACI tools."""

import json
import time
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path

import yaml
from jinja2 import Template

from tools import (
    ALL_COMMANDS, BASH_COMMAND, STR_REPLACE_EDITOR, SUBMIT_COMMAND,
    get_tool_schemas, get_command_docs,
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
                return self._make_result(patch, test_cmd, t0, step_idx + 1, errors)

        return self._make_result(self.env.get_patch(), test_cmd, t0, self.max_steps, errors)

    def _make_result(self, patch, test_cmd, t0, steps, errors):
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
            steps_taken=steps,
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

                if self._is_blocked(action):
                    self._requery("blocked", step, action)
                    continue

                if self._has_syntax_error(action):
                    self._requery("syntax", step, action)
                    continue

                break

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

        self.history.append({
            "role": "assistant",
            "content": step.output,
            "tool_calls": step.tool_calls,
        })

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
        if not action.strip().startswith(BASH_COMMAND.name):
            return False
        cmd = action.strip()[len(BASH_COMMAND.name):].strip().strip("'\"")
        result = subprocess.run(
            ["bash", "-n"], input=cmd, capture_output=True, text=True
        )
        return result.returncode != 0

    # --- Action execution ---

    def _execute_action(self, action: str) -> str:
        action = action.strip()
        assert self.env is not None

        bash_name = BASH_COMMAND.name
        editor_name = STR_REPLACE_EDITOR.name
        submit_name = SUBMIT_COMMAND.name

        if action.startswith(bash_name):
            cmd = action[len(bash_name):].strip().strip("'\"")
            return self.env.execute(cmd, timeout=self.execution_timeout)

        elif action.startswith(editor_name):
            return self._handle_str_replace_editor(action)

        elif action.startswith(submit_name):
            self.env.execute("git add -A", timeout=10)
            patch = self.env.get_patch()
            return f"<<SWE_AGENT_SUBMISSION>>\n{patch}"

        else:
            return self.env.execute(action, timeout=self.execution_timeout)

    def _handle_str_replace_editor(self, action: str) -> str:
        assert self.env is not None

        prefix = STR_REPLACE_EDITOR.name + " "
        args_str = action[len(prefix):].strip()

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
        marker = "<<SWE_AGENT_SUBMISSION>>"
        if marker in step.observation:
            patch = step.observation.split(marker, 1)[1].strip()
        else:
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
