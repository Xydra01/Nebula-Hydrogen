"""Wire the ``test.run`` / ``build.run`` / ``lint.run`` tools to the language adapters.

Each call resolves the target language (explicit ``language`` argument wins, else auto-detect from
the worktree), applies the worktree-confinement guard, dispatches to the adapter verb, and builds
the MCP response: a first content block with the compact structured result and a second with the
raw log (which the host spills to a blob when over the cap). ``isError`` reflects the outcome.
"""

from __future__ import annotations

import json
import os

from .adapters import errored_result
from .adapters.cpp_adapter import CppAdapter
from .adapters.python_adapter import PythonAdapter
from .adapters.rust_adapter import RustAdapter
from .detect import (
    ConfinementError,
    LanguageError,
    ensure_within_worktree,
    resolve_language,
    retired_drive_from_env,
)
from .mcp import Server, text_blocks
from .model import StructuredResult
from .run import RunOutput

_ADAPTERS = {
    "cpp": CppAdapter(),
    "python": PythonAdapter(),
    "rust": RustAdapter(),
}

_VERBS = ("test", "build", "lint")

_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "language": {
            "type": "string",
            "enum": ["cpp", "python", "rust"],
            "description": "Target language; if omitted, detected from the worktree.",
        },
        "cwd": {
            "type": "string",
            "description": "Directory to run in, relative to the worktree root; defaults to root.",
        },
    },
    "additionalProperties": False,
}


def _worktree_root() -> str:
    """The task's worktree root: the server's current working directory (set by the host)."""
    return os.getcwd()


def _response(result: StructuredResult, run: RunOutput) -> tuple[list[dict], bool]:
    """Build the MCP ``(content, is_error)``: structured result first, raw log second."""
    structured = json.dumps(result.to_dict(), separators=(",", ":"))
    blocks = [structured]
    raw = run.combined_log
    if raw:
        blocks.append(raw)
    return text_blocks(blocks), result.is_error


def _run_verb(verb: str, arguments: dict) -> tuple[list[dict], bool]:
    """Resolve language + confinement, dispatch to the adapter verb, build the response."""
    worktree_root = _worktree_root()
    retired = retired_drive_from_env()

    # Confinement: resolve the (optional) cwd within the worktree, off the retired drive.
    requested_cwd = arguments.get("cwd", ".")
    try:
        cwd = str(ensure_within_worktree(requested_cwd, worktree_root, retired))
    except ConfinementError as e:
        return _response(errored_result(verb, "unknown", str(e)), RunOutput(1, "", ""))

    # Language: explicit arg wins, else detect from the (confined) directory.
    try:
        language = resolve_language(arguments.get("language"), cwd)
    except LanguageError as e:
        return _response(errored_result(verb, "unknown", str(e)), RunOutput(1, "", ""))

    adapter = _ADAPTERS[language]
    method = getattr(adapter, verb)
    result, run = method(cwd)
    return _response(result, run)


def build_server() -> Server:
    """Construct the MCP server with the three build/test/lint tools registered."""
    server = Server()
    for verb in _VERBS:
        name = f"{verb}.run"
        description = {
            "test": "Run the test runner and return per-test pass/fail with file:line failures.",
            "build": "Run the build/compile step in the worktree and return compiler diagnostics.",
            "lint": "Run the linter in the worktree and return findings with file:line locations.",
        }[verb]

        # Bind `verb` per iteration so each handler dispatches to its own adapter method.
        def handler(arguments: dict, _verb: str = verb) -> tuple[list[dict], bool]:
            return _run_verb(_verb, arguments)

        server.register(name, description, _INPUT_SCHEMA, handler)
    return server
