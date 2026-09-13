"""ChatOpenAI subclass that preserves reasoning content from any OpenAI-compatible endpoint.

Why this exists
---------------
`langchain_openai.ChatOpenAI` targets the official OpenAI spec only: non-standard
response fields such as `reasoning_content` (DeepSeek, vLLM, DashScope/Qwen, GLM,
MiniMax, Moonshot, Groq, xAI, ...) or `reasoning` (vLLM >= its rename) are silently
dropped, so they never reach `AIMessage`.

But langchain-core can already render them: when
`AIMessage.additional_kwargs["reasoning_content"]` is set, the generic
`content_blocks` parsing lifts it into a `{"type": "reasoning", ...}` block.

One catch: ChatOpenAI tags every message with
`response_metadata["model_provider"] = "openai"`, and langchain-core then routes
`content_blocks` through its OpenAI translator, which only understands official
OpenAI fields and silently drops `reasoning_content`. (Verified: with the
"openai" tag `content_blocks` is `['text']`; with any other/absent tag it is
`['reasoning', 'text']`.) So this class also retags the message, which is the
same trick the official `ChatDeepSeek` uses with its "deepseek" tag.

The raw field is still kept verbatim in `additional_kwargs["reasoning_content"]`,
so nothing is lost even if you ignore `content_blocks`.

Usage
-----
    from oai_compat_chat import OAICompatChat

    llm = OAICompatChat(
        model="deepseek-flash",
        base_url="https://api.deepseek.com/v1",  # or any compatible gateway
        api_key="...",
    )
    msg = llm.invoke("...")
    msg.content_blocks     # -> [{'type': 'reasoning', ...}, {'type': 'text', ...}]
    msg.additional_kwargs  # -> {'reasoning_content': '...'}

Legacy (pre-`tools`) function calling
-------------------------------------
Endpoints that only accept `functions`/`function_call` still work: unknown kwargs
are merged into the JSON body untouched, so pass them directly instead of going
through `bind_tools()`:

    llm.invoke(messages, functions=[...], function_call="auto")
    llm.bind(functions=[...], function_call="auto")  # same, attached to the runnable

Metal logging (merged from 1_2_agent_wrapper_metal_log.py)
----------------------------------------------------------
Two opt-in switches, both off by default:

    llm = OAICompatChat(..., metal_log=True)                 # parsed OpenAI dicts
    llm = OAICompatChat(..., metal_log=True, raw_http=True)  # + literal HTTP bytes

`metal_log` dumps the OpenAI-format request payload (from `_get_request_payload`,
so it covers `invoke` and `stream` alike), the raw response body, and each
streamed chunk (at DEBUG level, since a stream can be long).

`raw_http` installs httpx event hooks that log method, URL, and body of every
request and every non-streaming response on the socket. Streaming responses log
status and URL only -- reading the body inside an httpx response hook would
consume the SSE stream before the caller can iterate it.

Copying reasoning into the next request
---------------------------------------
This class is stateless: it caches nothing between calls. The conversation lives
entirely in the message list *you* pass to each call (or in whatever memory /
checkpointer assembles it). So "round-tripping" here means only this: copy the
`reasoning_content` already sitting on the `AIMessage`s of the current call's
input into that same call's request body. If those messages are not in the list,
there is nothing to copy.

Copied by default: langchain-openai's `_convert_message_to_dict` serializes
`function_call`, `tool_calls`, and `audio` but never `reasoning_content`, so left
alone the field silently vanishes from the next request. This class puts it back.
Turn it off with `preserve_reasoning_to_request=False` if you would rather not
resend it -- every reasoning string is re-sent in full, so it costs prompt tokens.

    llm = OAICompatChat(..., reasoning_field="reasoning")   # OpenRouter / Anthropic key

Provider behavior differs:

- DeepSeek keys the whole thing off whether the *request* carries a `tools`
  parameter (not off whether a given past turn called a tool):
  <https://api-docs.deepseek.com/zh-cn/guides/thinking_mode>
    - request carries `tools`: every historical `reasoning_content` must be sent
      back and is concatenated into context. The docs state this holds even for
      turns that made no tool call, and that failing to return it earns a 400.
    - request carries no `tools`: prior `reasoning_content` is ignored, and is
      not concatenated into context even when it is present on the wire.
  So copying is essential for tool loops and a no-op for plain chat. Both probe
  scripts in this folder bind tools on every round, which puts them in the first
  case.

- OpenRouter / Anthropic-style gateways expect prior reasoning back
  (`reasoning` key) for tool-calling loops, where dropping it can break
  multi-turn tool use.

WARNING -- DeepSeek does not enforce the contract above, and silently does the
job for you. Read this before trusting `preserve_reasoning_to_request=False` as
a control condition, or before concluding anything from prompt-token counts.

  The docs require historical `reasoning_content` to be returned whenever the
  request carries `tools`, and state that omitting it yields a 400. Observed
  behaviour differs, in two ways (measured, not inferred, on repeated runs of
  `test_deepseek_thinking_preservation_by_langchain.py`):

  1. Omitting it does NOT produce the documented 400. With
     `preserve_reasoning_to_request=False` and `tools` bound, turn 3 returns
     HTTP 200 and answers correctly.
  2. What you *omit* is still present server-side, and billed. With the field
     stripped from a turn-3 payload of ~1.6 kB, the returned
     `usage.prompt_tokens` exceeded the payload's own token count by an amount
     matching the previous turn's `reasoning_tokens` 1:1 (ratios 1.006, 1.013,
     1.002 across three runs whose values spanned ~3x). In other words:
     `prompt_tokens ~= payload_tokens + omitted_reasoning_tokens`.

  The server does NOT simply substitute its own copy, though -- what the client
  sends still decides the outcome. Observed matrix for a turn-3 request in a
  tool loop (reasoning of the round-2 assistant message was variously intact,
  emptied, or overwritten with a bogus non-empty string, with the tool call id
  either left alone or mutated so it no longer matched its `ToolMessage`):

  | reasoning sent            | tool call id | round-3 behaviour | Contradiction with deepseek 20260912 docs? | 
  |---------------------------|--------------|-------------------| --- |
  | omitted / emptied         | intact       | acts as if it were there; answers with little or no visible derivation. | Yes, should have 400 instead |
  | bogus non-empty string    | intact       | corrupted context: wrong answer, or a long re-derivation. | No |
  | true text + junk prefix   | intact       | reads the answer off the supplied reasoning | No |
  | omitted / emptied         | mutated      | 400: "The `reasoning_content` in the thinking mode must be passed back to the API." | No |
  | bogus non-empty string    | mutated      | no 400, but a long re-derivation | No |

  Reading: an *omitted* field lets the request continue from the server's cached
  state for that turn, whereas any *substituted* value is taken at face value and
  derails the model. So the field is only silently supplied in the empty case --
  sending a placeholder is worse than sending nothing. The 400 fires when the
  tool-call association is broken and the field is missing, i.e. it validates the
  client's contract, and is bypassed only while the server can still match the
  turn.

  Practical consequences:

  - Setting `preserve_reasoning_to_request=False` does NOT remove the reasoning
    from DeepSeek's context; it only stops you from transmitting it. Such a run
    is an invalid control -- the model still sees the earlier thinking.
  - Reasoning can therefore influence a response and be billed without appearing
    anywhere in your request body. Do not use token accounting to decide whether
    a field was honoured; assume it may be counted even when you never sent it.
  - Never send a placeholder or truncated stand-in for reasoning: unlike an empty
    value, it is used verbatim and produces wrong answers. (This is also why a
    lossy truncation of reasoning is riskier than dropping it outright.)
  - Breaking the association (e.g. mutating a tool call's `id` so it no longer
    matches its `ToolMessage`) DOES trigger the documented error:
    `400 ... The reasoning_content in the thinking mode must be passed back to
    the API.` That error indicates the server lost track of the turn, not merely
    that this client omitted the field.
  - The silent continuation in the omitted case is suspected to ride on DeepSeek's
    context cache, which has a TTL. These measurements were all taken
    back-to-back; a slow loop with a long gap between turns may behave
    differently and fall back to the documented requirement.

  Keep `preserve_reasoning_to_request=True` (the default): sending the reasoning
  explicitly is the documented behaviour, and it is what makes behaviour
  provider-agnostic rather than dependent on this undocumented grace path.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import httpx
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_openai import ChatOpenAI
from loguru import logger
from pydantic import model_validator
import openai


def _extract_reasoning(source: Mapping[str, Any] | openai.BaseModel | None) -> str | None:
    """Read `reasoning_content` (or vLLM's newer `reasoning`) from a message/delta.

    Accepts the two shapes the wire actually produces: a plain dict, or an
    openai-SDK pydantic object. Pydantic v2 stores non-standard fields in
    `model_extra`, which some SDK versions hide from plain attribute access.

    Only non-empty `str` values are returned: an unexpected JSON type counts as
    "no reasoning" instead of leaking a non-str into `additional_kwargs`.
    """
    if source is None:
        return None
    if isinstance(source, Mapping):
        candidates: list[object] = [source.get("reasoning_content"), source.get("reasoning")]
        extra: object = source.get("model_extra")
    else:
        candidates = [getattr(source, "reasoning_content", None), getattr(source, "reasoning", None)]
        extra = getattr(source, "model_extra", None)
    if isinstance(extra, Mapping):
        candidates += [extra.get("reasoning_content"), extra.get("reasoning")]
    return next((text for text in candidates if isinstance(text, str) and text), None)


def _first_choice_field(
    payload: Mapping[str, Any] | openai.BaseModel,
    key: str,
) -> Mapping[str, Any] | openai.BaseModel | None:
    """Return `choices[0][key]`, i.e. the piece of a choice that carries reasoning.

    `key` is `"message"` for a full Chat Completions response and `"delta"` for a
    streamed chunk. Returns `None` unless the field is present *and* itself a dict
    or pydantic object -- exactly the shapes `_extract_reasoning` can read, so no
    unchecked JSON value ever reaches it.
    """
    choices = payload.get("choices") if isinstance(payload, Mapping) else getattr(payload, "choices", None)
    if not isinstance(choices, Sequence) or not choices:
        return None
    choice = choices[0]
    field = choice.get(key) if isinstance(choice, Mapping) else getattr(choice, key, None)
    return field if isinstance(field, (Mapping, openai.BaseModel)) else None


def _attach_reasoning(message: AIMessage | AIMessageChunk, text: str) -> None:
    """Attach reasoning to a message so both raw access and `content_blocks` work.

    Called per response, on the message object being returned to the caller --
    the model keeps no copy.

    - `additional_kwargs["reasoning_content"]`: the raw wire field, kept verbatim.
    - `response_metadata["model_provider"]`: retagged away from "openai". ChatOpenAI
      reports `model_provider="openai"` via `llm_output` (which langchain-core merges
      into the message *after* this hook runs), and that tag routes `content_blocks`
      through a translator that ignores non-standard reasoning fields. Any other tag
      falls back to the generic extractor, which understands `reasoning_content`.
    """
    message.additional_kwargs["reasoning_content"] = text
    message.response_metadata["model_provider"] = "openai-compatible"


# ── Metal logging helpers (from 1_2_agent_wrapper_metal_log.py) ──────────────
def _dump(label: str, obj: object) -> None:
    """Pretty-print a wire payload to loguru (`default=str` for odd values)."""
    logger.debug("── {} ──\n{}", label, json.dumps(obj, indent=2, ensure_ascii=False, default=str))


def _log_http_request(request: httpx.Request) -> None:
    logger.debug(
        ">>> HTTP {} {}\n{}",
        request.method,
        request.url,
        request.read().decode(errors="replace"),
    )


def _request_wants_stream(request: httpx.Request) -> bool:
    """True when the JSON body asks for `stream: true`."""
    try:
        return bool(json.loads(request.read()).get("stream"))
    except Exception:  # noqa: BLE001 - best-effort inspection only
        return False


def _log_http_response(response: httpx.Response) -> None:
    if _request_wants_stream(response.request):
        # Reading the body here would consume the SSE stream before the caller does.
        logger.debug("<<< HTTP {} {} (streaming response)", response.status_code, response.url)
        return
    logger.info(
        "<<< HTTP {} {}\n{}",
        response.status_code,
        response.url,
        response.read().decode(errors="replace"),
    )


def _raw_http_clients() -> tuple[httpx.Client, httpx.AsyncClient]:
    """Sync + async httpx clients wired to the socket-level loggers."""
    hooks = {"request": [_log_http_request], "response": [_log_http_response]}
    return httpx.Client(event_hooks=hooks), httpx.AsyncClient(event_hooks=hooks)


class OAICompatChat(ChatOpenAI):
    """ChatOpenAI + reasoning retention, safe against any OAI-compatible API.

    Set `metal_log=True` to dump the OpenAI request/response bodies, and
    `raw_http=True` to additionally log the literal bytes on the socket.
    """

    metal_log: bool = False
    """Dump the OpenAI-format request payload, response body, and stream chunks."""

    raw_http: bool = False
    """Log literal HTTP traffic via httpx event hooks."""

    preserve_reasoning_to_request: bool = True
    """Preserve `reasoning_content` from the `AIMessage`s in this call's input into the request body (default: on).

    WARNING: on DeepSeek, setting this to False does NOT make the model forget
    earlier reasoning, even though the API contract says it should error out.
    The server restores the omitted reasoning itself and still bills for it, so
    such a run is not a valid control condition. See the module docstring for
    the measurements.
    """

    reasoning_field: str = "reasoning_content"
    """Outbound key used when `preserve_reasoning_to_request` is on (e.g. `"reasoning"` for OpenRouter)."""

    @model_validator(mode="before")
    @classmethod
    def _install_raw_http_clients(cls, data: Any) -> Any:
        """Inject hook-wired httpx clients before the parent builds its SDK clients."""
        if isinstance(data, dict) and data.get("raw_http"):
            data = dict(data)
            if data.get("http_client") is None or data.get("http_async_client") is None:
                sync_client, async_client = _raw_http_clients()
                if data.get("http_client") is None:
                    data["http_client"] = sync_client
                if data.get("http_async_client") is None:
                    data["http_async_client"] = async_client
        return data

    def _preserve_reasoning_into_payload(self, input_: LanguageModelInput, payload: dict[str, Any]) -> None:
        """Preserve reasoning from this call's input `AIMessage`s into the outbound payload.

        Reads `additional_kwargs["reasoning_content"]` off the messages the caller
        passed in -- nothing is cached on the model, so a message only contributes
        if the caller included it in this call.

        Positional matching is used, and only when both sequences line up exactly;
        if the payload is not a Chat Completions one (e.g. the Responses API uses
        `input` instead of `messages`) or the lengths differ, nothing is written
        rather than risking reasoning attached to the wrong turn.
        """
        out_messages = payload.get("messages")
        if not isinstance(out_messages, list):
            return
        sources = self._convert_input(input_).to_messages()
        if len(sources) != len(out_messages):
            return
        for source, out in zip(sources, out_messages):
            if not isinstance(source, AIMessage) or not isinstance(out, dict):
                continue
            if out.get("role") != "assistant":
                continue
            reasoning = source.additional_kwargs.get("reasoning_content")
            if reasoning:
                out[self.reasoning_field] = reasoning

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Build the OpenAI-format body; mirrors `ChatOpenAI._get_request_payload`."""
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if self.preserve_reasoning_to_request:
            self._preserve_reasoning_into_payload(input_, payload)
        if self.metal_log:
            # Covers both invoke() and stream(), since both come through here.
            _dump("openai request body", payload)
        return payload

    def _create_chat_result(
        self,
        response: dict[str, Any] | openai.BaseModel,
        generation_info: dict[str, Any] | None = None,
    ) -> ChatResult:
        if self.metal_log:
            # Same union the parent accepts: a parsed SDK object, or a dict on error.
            raw = response if isinstance(response, dict) else response.model_dump()
            _dump("openai response body", raw)
        result = super()._create_chat_result(response, generation_info)
        reasoning = _extract_reasoning(_first_choice_field(response, "message"))
        if reasoning:
            for generation in result.generations:
                if isinstance(generation.message, (AIMessage, AIMessageChunk)):
                    _attach_reasoning(generation.message, reasoning)
        return result

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict[str, Any],
        default_chunk_class: type,
        base_generation_info: dict[str, Any] | None,
    ) -> ChatGenerationChunk | None:
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk, default_chunk_class, base_generation_info
        )
        if generation_chunk is not None and isinstance(generation_chunk.message, AIMessageChunk):
            if self.metal_log:
                logger.debug("── openai stream chunk ──\n{}", json.dumps(chunk, ensure_ascii=False, default=str))
            reasoning = _extract_reasoning(_first_choice_field(chunk, "delta"))
            if reasoning:
                _attach_reasoning(generation_chunk.message, reasoning)
        return generation_chunk
