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
        previous = history.pop()  # restore state before last edit
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
