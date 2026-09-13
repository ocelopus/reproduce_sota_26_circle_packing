"""
Filesystem tools for a bind_tools agent loop, built on the Deep Agents harness.

What the model sees, traced back to the source
----------------------------------------------
Every official tool reaches the model through exactly one chain:

    wrap_model_call()
      └─ _filter_unsupported_tools_and_apply_prompt(request)  # swaps grep desc, drops unsupported
           └─ FilesystemMiddleware.tools                      # [factory() for name, factory in tool_factories]
                └─ _create_<tool>_tool()                      # one factory per tool
                     └─ StructuredTool.from_function(
                            name=..., description=...,
                            func=sync_x, coroutine=async_x,
                            infer_schema=False, args_schema=XSchema)
                          └─ sync_x(args..., runtime)         # validate_path, permission check, format
                               └─ backend.method()            # the actual work

The structural fact this module is built on: **every official tool is a
`StructuredTool` whose `name`, `description`, `args_schema`, `func`, and
`coroutine` are plain readable attributes.** Nothing in the chain is
graph-dependent except a single parameter -- `runtime: ToolRuntime` -- which
exists so the body can read `runtime.tool_call_id` (`backend` is a closure
variable, not a runtime lookup). `ToolNode` normally injects that argument;
`_runtime()` below builds one directly. No other part of the chain is needed.

So there is nothing to reimplement. The borrowed core below is three things: a
`ToolRuntime`, a re-wrapper, and the factory loop. Everything else in this file is
an optional override.

The shell tool is the eighth factory of the same middleware
-----------------------------------------------------------
There is no shell middleware. `execute` comes out of the same
`FilesystemMiddleware.tools` list as the seven file tools:

    FilesystemMiddleware.tools
      └─ _create_execute_tool()               # same factory list as the file tools
           └─ StructuredTool.from_function(name="execute", description=..., func=sync_execute, ...)
                └─ sync_execute(command, runtime, timeout)
                     └─ backend.execute(command)   # subprocess.run(cmd, shell=True, cwd=root_dir)

The switch between the two tool sets is therefore the *backend class*, not any
middleware option:

| backend | tools | `execute` |
| --- | --- | --- |
| `FilesystemBackend` | 7 file tools | filtered out (backend is not a `SandboxBackendProtocol`) |
| `LocalShellBackend` | those 7 + `execute` | runs `subprocess.run(cmd, shell=True, cwd=root_dir)` on the host |

`LocalShellBackend` inherits `FilesystemBackend`, so one backend serves both sets
and `shell=True` below is just a constructor pick. Borrowing the shell tool needed
no new code: same `_runtime`, same `_borrow`.

One part does need re-borrowing. The `grep` and `execute` descriptions the
middleware creates are *provisional* -- they assume execution may be available --
and it reconciles them against the backend inside `wrap_model_call`. That hook is
unreachable from a `bind_tools` loop, so without help `grep` would tell the model
to "use the execute tool with `rg '<regex>'`" while no execute tool exists. The
factory therefore calls upstream's own reconcilers
(`_unsupported_tools_and_execution_state`, `_with_filtered_grep_description`,
`_with_filtered_execute_description`) in the order
`_filter_unsupported_tools_and_apply_prompt` calls them. Borrowed, not
reimplemented, and the self-test checks the result against that method itself.

Our own logic lives in `harness_tools_fixes.py` and arrives through a **single
import line** that can be deleted:

    from harness_tools_fixes import read_file_override, transforms  # ← DELETE THIS LINE

Both names are resolved lazily inside the factory, each guarded by the argument
that needs it:

| `done_right` | `default_read_limit` | name resolved | deleting the import |
| --- | --- | --- | --- |
| `False` | official default (100) | none | harmless -- the import is unused |
| `True` | official default | `transforms()` | `NameError` only if you asked for fixes |
| `False` | non-default | `read_file_override()` | `NameError` only if you asked for the override |
| `True` | non-default | both | `NameError` |

So the import is optional exactly when no fix is requested, and the error appears
exactly when one is. That is the property that keeps this file readable as pure
borrowing.

What still lives only in hooks, and is therefore out of reach here: token-based
eviction of oversized tool results (`wrap_tool_call`), `HumanMessage` eviction,
and `_route_host_path_prompt` (which returns `""` unless the backend is a
`CompositeBackend` whose default is a `LocalShellBackend` -- if you build that
stack, append it to your own system prompt).

Two more hook-only functions matter specifically for multimodal reads. Both are
module-level and take the whole message list, so they cannot be applied from a
per-`ToolMessage` fix; call them yourself before `invoke` if you need them:

| function | what it prevents | when it applies |
| --- | --- | --- |
| `_move_media_results_after_tool_results` | a synthetic media `HumanMessage` interleaved between the `ToolMessage`s of one tool-call batch, which providers reject | video reads only, and they need the optional extra |
| `_scrub_unsupported_multimodal_content` | a non-retryable 400 from a provider sent a block it cannot accept | a text-only model, or a non-PDF `file` block (e.g. `.docx`) on a provider outside the hard-coded OpenAI/Google allowlist |

The `read_file` media path itself needs neither in the common case -- blocks go
upstream as-is (`image_url` data URLs inside the tool message) and a
multimodal-capable endpoint accepts them. What has no cap at all is size: the
base64 payload of an image is built with no size or token check anywhere on the
read path (`max_file_size_mb` is consulted only by the Python grep fallback,
and token-based truncation is text-only and hook-only), and the block stays in
the message list, so it is re-sent in full on every later turn. The video path
is the exception: `MAX_VIDEO_INPUT_BYTES` and the extractor caps bound it, but
only when the optional video extra is installed. Without that extra `.mp4` is
still typed as a `video` block from the extension map and shipped as raw base64.

Upstream quirks we borrow as-is rather than patch, most likely to bite first:

1. `LocalShellBackend` pins `max_file_size_mb=10` in its `super().__init__()` and
   accepts no such argument, so `max_file_size_mb` is ignored when `shell=True`.
   (It only caps the Python grep fallback.)
2. Its shell environment defaults to EMPTY (`env=None, inherit_env=False`), so
   `git`, `python`, `node`... may not resolve for the model. Pass
   `shell_inherit_env=True`, or `shell_env={...}`, unless that is what you want.
3. `execute` ignores `virtual_mode`: the file tools speak virtual paths, the shell
   speaks host paths (`cwd=root_dir`), and nothing tells the model the mapping --
   the prompt section that would is composite-only (see above).
   `virtual_mode=False` makes both sides speak real paths.
4. `LocalShellBackend.execute` appends `\n\nExit code: N` on a non-zero exit and
   `_format_execute_output` then appends `[Command failed with exit code N]`: the
   exit code reaches the model twice.
5. `permissions=` and `shell=True` are mutually exclusive upstream: the middleware
   raises `NotImplementedError` ("Tool-level permissions for the execute tool are
   not implemented") for any shell-capable backend.

Usage
-----
    from harness_tools import make_delegating_tools

    tools = make_delegating_tools(root_dir="/path/to/repo")              # 7 file tools
    tools = make_delegating_tools(root_dir="/path/to/repo", shell=True)  # + execute

    tools_dict = {t.name: t for t in tools}

    llm_with_tools = llm.bind_tools(tools)
    results = [tools_dict[c["name"]].invoke(c) for c in response.tool_calls]

MIT-licensed upstream (`deepagents`); this module only wires its own objects up.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Callable, cast

from deepagents.backends import (
    DEFAULT_EXECUTE_TIMEOUT,
    BackendProtocol,
    FilesystemBackend,
    LocalShellBackend,
)
from deepagents.middleware.filesystem import (
    DEFAULT_READ_LIMIT,
    FilesystemMiddleware,
    FilesystemPermission,
)
from harness_tools_fixes import read_file_override, transforms  # ← DELETE THIS LINE: unused when both fix options are at their defaults; see the table in the module docstring.
from langchain.tools import ToolRuntime
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, StructuredTool

Fix = Callable[[ToolMessage], ToolMessage]
"""Rewrites a successful official `ToolMessage` in place of the model's view."""


# ── runtime ──────────────────────────────────────────────────────────────────
def _runtime(tool_call_id: str) -> ToolRuntime:
    """Build the minimal `ToolRuntime` the official tool bodies require.

    This is the *only* thing standing between `middleware.tools` and a plain
    `bind_tools` loop. Six fields are required; all are filled with inert values.
    `state` matters only for state-backed backends (`StateBackend`/`StoreBackend`),
    where files live in graph state -- `FilesystemBackend` reads and writes real
    disk and ignores it. `tool_call_id` is the one field the bodies actually read.
    `execute` reads it too: its capture-at-source offload keys off `tool_call_id`,
    and for `LocalShellBackend` (not a `BaseSandbox`) that offload is skipped.
    """
    return ToolRuntime(
        state={"messages": []},
        context=None,
        config={},
        tool_call_id=tool_call_id,
        stream_writer=lambda *a, **k: None,
        store=None,
    )


# ── the two functions that ARE the borrowing ────────────────────────────────
def _borrow(tool: BaseTool, fix: Fix | None = None) -> BaseTool:
    """Re-expose an official tool with `runtime` supplied, optionally fixing its output.

    `name`, `description`, and `args_schema` are carried over untouched, so the
    model sees exactly the official schema and prompt text. `tool_call_id` is
    injected via `InjectedToolCallId` and excluded from the schema, exactly as
    `ToolNode` would do it.
    """
    def shim(tool_call_id: Annotated[str, InjectedToolCallId], **kwargs: Any) -> Any:  # noqa: ANN401
        result = tool.func(**kwargs, runtime=_runtime(tool_call_id))  # type: ignore[union-attr]
        return fix(result) if fix and isinstance(result, ToolMessage) else result

    return StructuredTool.from_function(
        func=shim,
        name=tool.name,
        description=tool.description,
        args_schema=tool.args_schema,
    )


def make_delegating_tools(
    root_dir: str | Path | None = None,
    *,
    shell: bool = False,
    shell_timeout: int = DEFAULT_EXECUTE_TIMEOUT,
    shell_max_output_bytes: int = 100_000,
    shell_env: dict[str, str] | None = None,
    shell_inherit_env: bool = False,
    max_execute_timeout: int = 3600,
    backend: BackendProtocol | None = None,
    virtual_mode: bool = True,
    max_file_size_mb: int = 50,
    permissions: list[FilesystemPermission] | None = None,
    done_right: bool = False,
    default_read_limit: int = DEFAULT_READ_LIMIT,
    exclude: tuple[str, ...] = (),
) -> list[BaseTool]:
    """The official filesystem and shell tools, usable in a plain `bind_tools` loop.

    Every tool body is upstream's: validation, permission checks, formatting, and
    error strings all come from `deepagents.middleware.filesystem` by
    construction, so there is no string to drift and nothing to fix when upstream
    rewords. Tool names are the official `read_file`, `write_file`, `edit_file`,
    `ls`, `glob`, `grep`, `delete`, plus `execute` when a shell is enabled.

    Args:
        root_dir: Directory every path resolves against, and the shell's working
            directory. In `virtual_mode` the model sees paths *relative to this
            root* (`/module_1/foo.py`), never the real absolute path, so the
            sandbox boundary is not leaked back into the transcript.
        shell: Borrow `execute` too, by building a `LocalShellBackend` instead of a
            `FilesystemBackend`. Same middleware, same factory list; the backend
            class is the only switch. Upstream's own warning applies: this gives
            the model unrestricted shell execution on the host, and `virtual_mode`
            does not restrict it (quirk 3).
        shell_timeout: `LocalShellBackend`'s default per-command timeout, in seconds
            (upstream default 120). The model can override it per call with the
            tool's `timeout` argument, capped by `max_execute_timeout`.
        shell_max_output_bytes: Output captured before truncation (upstream default
            100_000). The truncation note is added by the backend, not the tool.
        shell_env: Environment for shell commands. `None` (upstream default) means
            an EMPTY environment unless `shell_inherit_env` is set -- see quirk 2.
        shell_inherit_env: Start from this process's environment, then apply
            `shell_env` overrides on top.
        max_execute_timeout: Upper bound the middleware accepts for the model's
            per-command `timeout`; anything larger comes back as a tool error.
        backend: Pass your own initialized backend to win over all of the above --
            e.g. a `BaseSandbox` subclass (Docker, remote host) for a real sandbox,
            or a `CompositeBackend`. `execute` appears iff the backend satisfies
            upstream's `supports_execution` predicate, decided below by upstream's
            own capability filter, never by name.
        virtual_mode: When `True` (recommended), block traversal and anything
            resolving outside `root_dir` for the *file* tools. Set `False` only for
            trusted local work that genuinely needs unrestricted host paths, or to
            make file paths and shell paths agree (quirk 3).
        max_file_size_mb: Cap for the Python grep fallback (ripgrep has its own).
            Ignored when `shell=True`; see quirk 1.
        permissions: Official `FilesystemPermission` rules. These DO apply here --
            `_check_fs_permission` runs inside each tool body, not in a hook. Not
            combinable with `shell=True` upstream; see quirk 5.
        done_right: Apply `harness_tools_fixes.transforms()` to correct three
            upstream output quirks: `ls`/`glob` leaking `str(list)`, `read_file`
            not marking end-of-file, and `grep` explaining a regex-shaped
            no-match. (`read_file`'s multimodal blocks are left alone: they are
            already the correct form, and an earlier refusal in this module's
            history destroyed image reads.) Off by default, so `True`/`False` is
            a clean A/B of model-visible text.
        default_read_limit: Lines `read_file` returns when unspecified. Applied via
            `harness_tools_fixes.read_file_override()`, which rebuilds the schema
            default and the description text together from upstream parts so they
            cannot disagree. Passing `DEFAULT_READ_LIMIT` skips that call entirely.
        exclude: Official tool names to omit entirely, applied before the capability
            check. Empty by default: `execute` is not "excluded", it is absent
            exactly when the backend cannot serve it -- the same rule upstream
            applies per request. Use `exclude=("execute",)` to keep a
            shell-capable backend but hide the tool; the `grep` description then
            reverts to its no-execute variant.

    Returns:
        The official tools, re-wrapped, in upstream's factory order, with the
        model-visible `(name, description)` pairs equal to what upstream's own
        request-time filter produces for the same backend -- and schemas upstream's
        byte for byte, apart from `read_file` when `default_read_limit` differs.
    """
    if backend is None:
        backend = (
            LocalShellBackend(
                root_dir=root_dir,
                virtual_mode=virtual_mode,
                timeout=shell_timeout,
                max_output_bytes=shell_max_output_bytes,
                env=shell_env,
                inherit_env=shell_inherit_env,
            )
            if shell
            else FilesystemBackend(
                root_dir=root_dir,
                virtual_mode=virtual_mode,
                max_file_size_mb=max_file_size_mb,
            )
        )

    middleware = FilesystemMiddleware(
        backend=backend,
        max_execute_timeout=max_execute_timeout,
        _permissions=permissions,
    )

    candidates: list[StructuredTool] = []
    for candidate in middleware.tools:
        # `middleware.tools` is typed loosely (`BaseTool | dict`), but every entry
        # is a `StructuredTool` built by a `_create_*_tool` factory.
        inner = cast("StructuredTool", candidate)
        if inner.name not in exclude:
            candidates.append(inner)

    # The three official calls `_filter_unsupported_tools_and_apply_prompt` makes,
    # on our tool list instead of on a `ModelRequest`: drop tools the backend
    # cannot serve (`execute` without a shell, `delete` without backend support),
    # then reword `grep` and `execute` for that answer.
    names: set[str] = {tool.name for tool in candidates}
    # Upstream annotates the parameter as `set[str | None]` because a request may
    # carry unnamed tool dicts; ours are all named tools.
    unsupported, execution_active, _ = middleware._unsupported_tools_and_execution_state(cast("set[str | None]", names))
    visible: list[BaseTool | dict[str, Any]] = [tool for tool in candidates if tool.name not in unsupported]
    visible = middleware._with_filtered_grep_description(visible, include_execution=execution_active)
    visible = middleware._with_filtered_execute_description(
        visible,
        visible_search_tools={name for name in names if name not in unsupported},
    )

    # Both fix lookups are lazy and individually guarded, so deleting the
    # `harness_tools_fixes` import above is harmless unless a fix is requested.
    fixes: dict[str, Fix] = transforms() if done_right else {}
    override_read = default_read_limit != DEFAULT_READ_LIMIT

    tools: list[BaseTool] = []
    for candidate in visible:
        inner = cast("StructuredTool", candidate)
        if inner.name == "read_file" and override_read:
            # Only `read_file` needs rebuilding, and only when its default moves:
            # the schema and the description both state the number.
            schema, description = read_file_override(default_read_limit)
            inner = StructuredTool.from_function(
                func=inner.func,  # type: ignore[arg-type]
                coroutine=inner.coroutine,  # type: ignore[arg-type]
                name=inner.name,
                description=description,
                args_schema=schema,
            )
        tools.append(_borrow(inner, fixes.get(inner.name)))
    return tools


if __name__ == "__main__":
    import tempfile
    from typing import cast

    from deepagents.middleware.filesystem import EXECUTE_TOOL_DESCRIPTION, GREP_TOOL_DESCRIPTION

    # Self-test. Run directly (`python harness_tools.py`) after any `deepagents`
    # upgrade. Every assertion below is a property this module claims, and the
    # upstream surface it depends on is public but not documented, so this is the
    # canary for a silent change.
    with tempfile.TemporaryDirectory(prefix="harness_tools_") as tmp:
        root = Path(tmp)
        (root / "pkg").mkdir()
        (root / "pkg" / "mod.py").write_text(
            "def compute(x):\n    total = x + 1\n    return total\n\n\ndef other():\n    return 2\n"
        )
        (root / "blob.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x01\x02")
        (root / "empty.txt").write_text("")
        (root / "big.txt").write_text("\n".join(f"line {i}" for i in range(400)))

        def make(**kwargs: Any) -> dict[str, BaseTool]:
            # `shell_inherit_env=True` so the shell tests get a PATH (quirk 2).
            return {t.name: t for t in make_delegating_tools(root, **kwargs)}

        plain = make()
        fixed = make(done_right=True)
        shell_tools = make(shell=True, shell_inherit_env=True)

        class _StubRequest:
            """Duck-typed `ModelRequest` for the filter path.

            `_filter_unsupported_tools_and_apply_prompt` only reads `.tools` and
            calls `.override(tools=...)` for these backends: no system prompt is
            assembled (the host-path routing section is composite-only), and no
            other hook is reached.
            """

            def __init__(self, tools: list[Any]) -> None:
                self.tools = tools
                self.system_message = None

            def override(self, **changes: Any) -> "_StubRequest":
                new = _StubRequest(changes.get("tools", self.tools))
                new.system_message = self.system_message
                return new

        def model_visible(backend: BackendProtocol) -> dict[str, BaseTool]:
            """What upstream would put in the request for this backend, via its own filter."""
            middleware = FilesystemMiddleware(backend=backend)
            # Duck-typed on purpose: the filter path reads `.tools` and calls
            # `.override(tools=...)`, nothing else, and building a real
            # `ModelRequest` would need a model just to test string equality.
            request = cast("Any", _StubRequest(list(middleware.tools)))
            filtered = middleware._filter_unsupported_tools_and_apply_prompt(request)
            return {cast("BaseTool", tool).name: cast("BaseTool", tool) for tool in filtered.tools}

        def invoke(registry: dict[str, BaseTool], name: str, **args: Any) -> ToolMessage:
            out = registry[name].invoke(
                {"name": name, "args": args, "id": f"call_{name}", "type": "tool_call"}
            )
            return cast("ToolMessage", out if not isinstance(out, list) else out[-1])

        def body(registry: dict[str, BaseTool], name: str, **args: Any) -> str:
            return str(invoke(registry, name, **args).content)

        print("=== names: execute is present iff the backend can serve it ===")
        print("   plain (FilesystemBackend):", sorted(plain))
        print("   shell (LocalShellBackend):", sorted(shell_tools))
        print("   plain has no execute:", "execute" not in plain, "| shell has execute:", "execute" in shell_tools)

        print("\n=== what the model sees == upstream's request-time filter ===")
        # The strongest claim this module makes, checked against the official
        # method itself rather than against a copy of its strings.
        for label, ours, backend in (
            ("plain", plain, FilesystemBackend(root_dir=root)),
            ("shell", shell_tools, LocalShellBackend(root_dir=root)),
        ):
            theirs = model_visible(backend)
            names_equal = sorted(ours) == sorted(theirs)
            desc_equal = all(ours[n].description == theirs[n].description for n in ours)
            schema_equal = all(
                cast("Any", ours[n].tool_call_schema).model_json_schema()
                == cast("Any", theirs[n].tool_call_schema).model_json_schema()
                for n in ours
            )
            print(f"   {label:5s} names={names_equal} descriptions={desc_equal} schemas={schema_equal} ({len(ours)} tools)")

        print("\n=== the two reconciled descriptions, pinned to upstream strings ===")
        print("   shell execute == EXECUTE_TOOL_DESCRIPTION:", shell_tools["execute"].description == EXECUTE_TOOL_DESCRIPTION)
        print("   shell grep    == GREP_TOOL_DESCRIPTION   :", shell_tools["grep"].description == GREP_TOOL_DESCRIPTION)
        print("   plain grep has no execute mention        :", "execute" not in plain["grep"].description)
        print("   plain grep != with-execute variant       :", plain["grep"].description != GREP_TOOL_DESCRIPTION)

        print("\n=== execute runs commands (shell mode) ===")
        out = invoke(shell_tools, "execute", command="pwd")
        cwd_reported = str(out.content).splitlines()[0]
        print(f"   cwd = root_dir     : {cwd_reported!r} == {str(root)!r} -> {cwd_reported == str(root)}")
        out = invoke(shell_tools, "execute", command="echo out; echo err 1>&2")
        print("   stdout+stderr      :", "[stderr] err" in str(out.content), "|", str(out.content).splitlines())
        out = invoke(shell_tools, "execute", command="exit 3")
        print(
            "   failed exit        :",
            "[Command failed with exit code 3]" in str(out.content),
            "| reported twice (quirk 4):",
            "Exit code: 3" in str(out.content),
        )
        out = invoke(shell_tools, "execute", command="sleep 5", timeout=1)
        print(
            "   per-command timeout:",
            "timed out after 1 seconds" in str(out.content),
            "| exit 124 surfaced:",
            "exit code 124" in str(out.content),
        )
        out = invoke(shell_tools, "execute", command="true", timeout=-1)
        print(f"   negative timeout   : status={out.status!r} {str(out.content)!r}")
        out = invoke(shell_tools, "execute", command="true", timeout=99999)
        print(f"   over max timeout   : status={out.status!r} {str(out.content)!r}")
        out = invoke(shell_tools, "execute", command="echo hi")
        print(f"   artifact           : {out.artifact!r} | status={out.status!r} | id={out.tool_call_id!r}")

        tight = make(shell=True, shell_inherit_env=True, max_execute_timeout=10)
        out = invoke(tight, "execute", command="true", timeout=11)
        print("   max_execute_timeout=10 passthrough:", str(out.content))

        print("\n=== exclude still hides execute from a shell-capable backend ===")
        hidden = make(shell=True, shell_inherit_env=True, exclude=("execute",))
        print(
            "   execute hidden:",
            "execute" not in hidden,
            "| grep reverts to no-execute description:",
            "execute" not in hidden["grep"].description,
        )

        print("\n=== backend= escape hatch wins over shell= ===")
        own_shell = {t.name: t for t in make_delegating_tools(backend=LocalShellBackend(root_dir=root, inherit_env=True))}
        own_fs = {t.name: t for t in make_delegating_tools(backend=FilesystemBackend(root_dir=root))}
        print(
            "   own LocalShellBackend:",
            "execute" in own_shell,
            f"({len(own_shell)} tools)",
            "| own FilesystemBackend:",
            "execute" not in own_fs,
            f"({len(own_fs)} tools)",
        )

        print("\n=== permissions + shell: upstream refuses at construction ===")
        try:
            make(shell=True, permissions=[FilesystemPermission(operations=["read"], paths=["/pkg/**"], mode="deny")])
            print("   BAD: expected NotImplementedError")
        except NotImplementedError as exc:
            print(f"   ok  NotImplementedError: {exc}")

        print("\n=== status + tool_call_id on both paths ===")
        cases: list[tuple[str, str, dict[str, Any]]] = [
            ("read_file", "success", {"file_path": "/pkg/mod.py", "limit": 3}),
            ("read_file", "error", {"file_path": "/nope.txt"}),
            ("read_file", "error", {"file_path": "/../../etc/passwd"}),
            ("read_file", "error", {"file_path": "C:/Users/x.txt"}),
            ("edit_file", "error", {"file_path": "/pkg/mod.py", "old_string": "zz", "new_string": "q"}),
            ("ls", "error", {"path": "/nope"}),
            ("glob", "error", {"pattern": "../*"}),
            ("delete", "error", {"file_path": "/nope.txt"}),
        ]
        for name, want, args in cases:
            out = invoke(plain, name, **args)
            flag = "ok " if out.status == want and out.tool_call_id == f"call_{name}" else "BAD"
            print(f"   {flag} {name:10s} status={out.status!r:9s} id={out.tool_call_id!r:16s} {str(out.content)[:52]!r}")

        print("\n=== done_right: the output corrections ===")
        for label, name, args in [
            ("ls", "ls", {"path": "/"}),
            ("glob", "glob", {"pattern": "*.py"}),
            ("read (window)", "read_file", {"file_path": "/big.txt", "limit": 5}),
            ("read (whole)", "read_file", {"file_path": "/big.txt", "limit": 1000}),
            ("read (media)", "read_file", {"file_path": "/blob.png"}),
            ("grep (no match)", "grep", {"pattern": "zzz"}),
        ]:
            print(f"-- {label} --")
            print("   official  :", repr(body(plain, name, **args)[:100]))
            print("   done_right:", repr(body(fixed, name, **args)[:100]))
            if label == "read (whole)":
                # The correction is at the tail, so the truncated prints above look
                # identical. Show the end explicitly.
                for tag, reg in (("official  ", plain), ("done_right", fixed)):
                    print(f"   {tag} tail: {body(reg, name, **args)[-30:]!r}")

        print("\n=== a media read survives done_right intact ===")
        # The regression this guards: `read_file_done_right` used to rewrite any
        # non-`str` content into "is binary ... returns text only", which turned
        # every image read into a tool error under `done_right=True`.
        for tag, registry in (("official  ", plain), ("done_right", fixed)):
            out = invoke(registry, "read_file", file_path="/blob.png")
            blocks = out.content if isinstance(out.content, list) else []
            shape = [(b.get("type"), b.get("mime_type"), len(b.get("base64", ""))) for b in blocks]
            print(f"   {tag}: status={out.status} type={type(out.content).__name__} blocks={shape}")
        print("   identical:", invoke(plain, "read_file", file_path="/blob.png").content
              == invoke(fixed, "read_file", file_path="/blob.png").content)

        print("\n=== done_right leaves errors and normal content alone ===")
        for name, args in [
            ("read_file", {"file_path": "/nope.txt"}),
            ("read_file", {"file_path": "/empty.txt"}),
            ("read_file", {"file_path": "/pkg/mod.py"}),
            ("read_file", {"file_path": "/blob.png"}),
            ("grep", {"pattern": "return"}),
        ]:
            same = body(plain, name, **args) == body(fixed, name, **args)
            print(f"   {name:10s} {str(args)[:44]:46s} unchanged={same}")
        print("   (a full successful text read is the one intentional exception: the end-of-file marker)")
        print("   (a full successful read is the one intentional exception: the add-on end-of-file marker)")

        print("\n=== permissions are enforced in-body ===")
        denied = {t.name: t for t in make_delegating_tools(
            root,
            permissions=[FilesystemPermission(operations=["read"], paths=["/pkg/**"], mode="deny")],
        )}
        out = invoke(denied, "read_file", file_path="/pkg/mod.py")
        print(f"   {out.content!r} | status={out.status!r}")

        print("\n=== default_read_limit moves schema and description together ===")
        tuned = {t.name: t for t in make_delegating_tools(root, default_read_limit=400)}

        def limit_default(tool: BaseTool) -> Any:
            fields = getattr(tool.args_schema, "model_fields")
            return fields["limit"].default

        print("   schema default        :", limit_default(tuned["read_file"]))
        print("   description says 400  :", "up to 400 lines" in tuned["read_file"].description)
        print("   official default kept :", limit_default(plain["read_file"]))
        print("   other tools untouched :", tuned["ls"].args_schema is plain["ls"].args_schema)

        print("\n=== deleting the fix import: which combinations survive? ===")
        # Deleting the two names from this module's namespace is exactly what
        # deleting the import line does -- the import's only effect is binding
        # those names here. So this tests the real property in-process.
        import harness_tools as self_module

        saved = (self_module.transforms, self_module.read_file_override)
        del self_module.transforms
        del self_module.read_file_override
        expectations: list[tuple[bool, bool, bool]] = [
            # (done_right, non-default limit, can_delete_import)
            (False, False, True),
            (True, False, False),
            (False, True, False),
            (True, True, False),
        ]
        try:
            for want_fixes, want_override, should_work in expectations:
                kwargs: dict[str, Any] = {
                    "done_right": want_fixes,
                    "default_read_limit": 400 if want_override else DEFAULT_READ_LIMIT,
                }
                try:
                    self_module.make_delegating_tools(root, **kwargs)
                    outcome, detail = True, "built fine"
                except NameError as exc:
                    outcome, detail = False, f"NameError: {exc}"
                flag = "ok " if outcome == should_work else "BAD"
                label = f"done_right={want_fixes!s:5s} limit={'400' if want_override else '100'}"
                print(f"   {flag} {label}  works={outcome!s:5s} ({detail})")
        finally:
            self_module.transforms, self_module.read_file_override = saved

        print("\n--- final content ---")
        print((root / "pkg" / "mod.py").read_text())
