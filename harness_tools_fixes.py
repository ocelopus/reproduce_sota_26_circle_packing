"""
Everything we add on top of the borrowed Deep Agents filesystem tools.

`make_delegating_tools` runs the official tool bodies verbatim, so this module
holds *only* our own logic. That boundary is deliberate: delete the single
`from harness_tools_fixes import ...` line in `harness_tools.py` and the borrowing
is pure, with no residue from us. Every name here is resolved lazily inside the
factory and guarded by the option that needs it, so an unnecessary import costs
nothing:

| `done_right` | `default_read_limit` | name resolved | effect of deleting the import |
| --- | --- | --- | --- |
| `False` | official default (100) | none | nothing -- the import is simply unused |
| `True` | official default | `transforms()` | `NameError` only if you asked for fixes |
| `False` | non-default | `read_file_override()` | `NameError` only if you asked for the override |
| `True` | non-default | both | `NameError` |

So the import is optional exactly when no fix is requested, and the error appears
exactly when one is.

Two kinds of fix live here, both triggered by explicit arguments rather than being
unconditionally applied:

- **Output corrections** (`transforms`), for three official tools whose output is
  awkward at a hand-rolled tool boundary.
- **A config override** (`read_file_override`), for the official `read_file`
  schema and description, which hard-code the default read limit as a literal.

Used by `harness_tools.make_delegating_tools`. These corrections apply to the file
tools in both modes (`shell=True` or not); `execute` output has no entry here
because upstream formats it in `_format_execute_output` and we borrow that as-is.
"""

from __future__ import annotations

import ast
import re
from typing import Any, Callable

from deepagents.backends.utils import EMPTY_CONTENT_WARNING
from deepagents.middleware.filesystem import (
    _READ_FILE_TOOL_DESCRIPTION_TEMPLATE,
    ReadFileSchema,
)
from langchain_core.messages import ToolMessage
from pydantic import Field, create_model

Fix = Callable[[ToolMessage], ToolMessage]

_READ_NOTICE_RE = re.compile(r"\[Read \d+ lines? \(lines \d+-\d+")
"""Matches the official pagination notice, in both of its two variants.

`_remaining_lines_notice` emits `... lines remaining from offset N.` when
`total_lines` is known and `More lines remain from offset N.` when it is not, so
neither suffix alone is sufficient. The shared `[Read N lines (lines X-Y` prefix
is, and it does not collide with `_clamped_offset_notice`, which opens with
`[Requested offset`.
"""

_NO_LINES_MARKER = "no lines were read because"
"""Substring of the official `NO_LINES_REQUESTED_WARNING` (its `limit` is formatted in)."""


def paths_done_right(message: ToolMessage) -> ToolMessage:
    """Join `ls` / `glob` paths with newlines instead of leaking a Python list.

    Official runs its path list through `truncate_if_too_long`, which returns a
    `list[str]`, and the caller then stringifies it -- so the model receives
    `"['/a.py', '/b.py']"`, complete with quotes and list syntax. The truncation
    budget is worth keeping, so the official string is parsed back rather than
    recomputed: `ast.literal_eval` recovers the exact list, including the
    truncation guidance entry when present.
    """
    if message.status != "success":
        return message
    try:
        paths = ast.literal_eval(str(message.content))
    except (ValueError, SyntaxError):
        return message  # never expected, but a failed parse must not lose the answer
    if not isinstance(paths, list):
        return message
    return message.model_copy(update={"content": "\n".join(str(p) for p in paths)})


def read_file_done_right(message: ToolMessage) -> ToolMessage:
    """Mark the end of file explicitly. Multimodal reads pass through untouched.

    - **Media.** For an image, audio, video, or PDF, `message.content` is a
      `list` of content blocks -- base64 plus a MIME type, which is exactly what
      a multimodal model needs. This function used to refuse every non-`str`
      result with "is binary ... this tool returns text only", to suppress a
      *different* bug: an earlier hand-written `read_file` of ours line-numbered
      raw base64 as if it were source text. The official tool never does that,
      so the refusal outlived the bug it was written for and came to do nothing
      but destroy image reads -- the one output where refusing is worst.

      Provider-level rejection of a block is *not* handled here: upstream's
      `_scrub_unsupported_multimodal_content` replaces unsupported blocks with a
      text placeholder, and it runs in a `wrap_model_call` hook this module does
      not have. See the note in `harness_tools`.
    - **End of file.** Official's footer is silent once the window reaches EOF,
      so an absent footer has to be read as "you have the whole file" -- while an
      absent footer for any other reason looks identical. A completion marker
      removes the ambiguity. Detected by exclusion: a successful message that is
      text, is not one of the two system reminders, and carries no pagination
      notice can only be a read that reached the end.
    """
    if message.status != "success":
        return message

    if not isinstance(message.content, str):
        return message  # multimodal blocks: already the correct model-facing form

    if _READ_NOTICE_RE.search(message.content):
        return message  # a partial window; the official notice already says so
    if EMPTY_CONTENT_WARNING in message.content or _NO_LINES_MARKER in message.content:
        return message  # a system reminder, not file content
    return message.model_copy(update={"content": f"{message.content}\n\n[End of file.]"})


def grep_done_right(message: ToolMessage) -> ToolMessage:
    """Explain a regex-shaped pattern that matched nothing.

    The pattern is matched literally, so `foo|bar` finds nothing and the plain
    `No matches found` reads as "not in the codebase". Saying why costs one line
    and prevents a wrong conclusion.
    """
    if message.status != "success" or message.content != "No matches found":
        return message
    return message.model_copy(
        update={
            "content": (
                "No matches found. Note: the pattern is matched literally, not as a "
                "regular expression, so characters like `|`, `.*`, and `\\.` are "
                "ordinary text. For `|` alternation, search for each alternative "
                "separately."
            )
        }
    )


def transforms() -> dict[str, Fix]:
    """The output fixes, keyed by the official tool name they apply to.

    Returned as a fresh dict rather than a module constant so that importing this
    module never builds anything at import time.
    """
    return {
        "ls": paths_done_right,
        "glob": paths_done_right,
        "read_file": read_file_done_right,
        "grep": grep_done_right,
    }


def read_file_override(default_read_limit: int) -> tuple[Any, str]:
    """Rebuild the official `read_file` schema default and description together.

    Upstream hard-codes the read limit as a literal in two places at once: the
    `limit` field's default in `ReadFileSchema`, and the sentence "by default, it
    reads up to 100 lines" in the description template. Changing one without the
    other puts a false promise in the prompt, so both are derived here from the
    upstream parts -- the schema by subclassing `ReadFileSchema` and overriding a
    single field, the description from the same template upstream formats.

    Returns:
        `(schema, description)` to hand to `StructuredTool.from_function`.
    """
    field = ReadFileSchema.model_fields["limit"]
    schema = create_model(
        "ReadFileSchemaWithLimit",
        __base__=ReadFileSchema,
        limit=(int, Field(default=default_read_limit, description=field.description)),
    )
    description = _READ_FILE_TOOL_DESCRIPTION_TEMPLATE.format(
        first_line=(
            f"By default, it reads up to {default_read_limit} lines starting "
            "from the beginning of the file"
        ),
        multimodal_bullets="",
    )
    return schema, description
