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
