"""
Protocol layer shared by the OpenAI-compatible backends.

Each backend module owns exactly one deployment — its endpoint, its credentials,
its model ids — and none of them import each other:

    openai_gpt.py   api.openai.com        Responses API + the async Batch API
    codex_gpt.py    LiteLLM/Codex proxy   Responses API, no Batch API

What they share is the wire protocol, not the deployment:

    OpenAIWireMixin    message / tool / structured-output conversions. Stateless
                       — reads `self.model_name`, `self.enable_thinking` and
                       `self.reasoning_effort` from the class mixing it in.
    ResponsesAPIMixin  the Responses API call path built on those conversions.
"""
from __future__ import annotations

import base64
import copy
import json
import logging
import re
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

from openai import BadRequestError
from PIL import Image
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt, wait_fixed

from models.vllm_client import (
    _ThinkingBlock,
    _TextBlock,
    _ToolUseBlock,
    _UsageAdapter,
    _ResponseAdapter,
)

logger = logging.getLogger(__name__)

__all__ = ["OpenAIWireMixin", "ResponsesAPIMixin", "_StreamedResp", "_batch_error_msg",
           "_EFFORT_THINKING", "_EFFORT_NON_THINKING",
           "_ThinkingBlock", "_TextBlock", "_ToolUseBlock",
           "_UsageAdapter", "_ResponseAdapter"]


def _batch_error_msg(obj: dict) -> str:
    """Pull a human-readable error message out of one batch output/error line."""
    err = obj.get("error")
    if not err:
        body = (obj.get("response") or {}).get("body") or {}
        err = body.get("error")
    if isinstance(err, dict):
        return str(err.get("message") or err)[:500]
    if err:
        return str(err)[:500]
    return str(obj)[:500]


# enable_thinking -> reasoning_effort mapping.
# Note: "minimal" is only valid on some GPT-5 models (e.g. 5.5); gpt-5.4-mini
# supports none/low/medium/high/xhigh, so use "none" for the non-thinking path.
_EFFORT_THINKING     = "high"
_EFFORT_NON_THINKING = "none"


class _StreamedResp:
    """Reconstructed Responses result from a STREAMED call. Needed because some
    proxies (LiteLLM :4000) return status=completed with an EMPTY output[] on a
    non-streaming call; the real items only arrive as stream events. Exposes the
    same .output / .usage / .output_text a normal response does."""
    def __init__(self, output, usage, output_text):
        self.output = output
        self.usage = usage
        self.output_text = output_text



class OpenAIWireMixin:
    """Message / tool / structured-output conversions for the OpenAI wire format."""

    def _resolve_reasoning_effort(
        self,
        *,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
    ) -> str:
        thinking_flag = self.enable_thinking if enable_thinking is None else enable_thinking
        if self.model_name == "gpt-5":
            default_effort = "minimal"
        else:
            default_effort = _EFFORT_THINKING if thinking_flag else _EFFORT_NON_THINKING
        return reasoning_effort or self.reasoning_effort or default_effort

    # ------------------------------------------------------------------
    # Message → OpenAI multimodal format (same shape as VLLMClient)
    # ------------------------------------------------------------------
    def _encode_image(self, image: Image.Image) -> str:
        buf = BytesIO()
        image.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _to_openai_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        result = []
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            if isinstance(content, str):
                result.append({"role": role, "content": content})
            elif isinstance(content, list):
                parts = []
                for item in content:
                    if item.get("type") == "text":
                        parts.append({"type": "input_text", "text": item["text"]})
                    elif item.get("type") == "image" and isinstance(item.get("image"), Image.Image):
                        b64 = self._encode_image(item["image"])
                        parts.append({
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{b64}",
                        })
                result.append({"role": role, "content": parts})
            else:
                result.append({"role": role, "content": str(content)})
        return result

    @staticmethod
    def _convert_tools_to_openai(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                },
            }
            for t in tools
        ]

    @staticmethod
    def _convert_tools_to_responses(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [
            {
                "type": "function",
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
                "strict": False,
            }
            for t in tools
        ]

    @staticmethod
    def _get_block_attr(block: Any, attr: str, default: Any = None) -> Any:
        if isinstance(block, dict):
            return block.get(attr, default)
        return getattr(block, attr, default)

    def _messages_to_responses_input_simple(
        self, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Convert plain text+image messages to Responses API input format (no tool turns).

        System-role messages are skipped — pass them via instructions= instead.
        """
        result = []
        for msg in messages:
            role = msg["role"]
            if role == "system":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                result.append({
                    "type": "message",
                    "role": role,
                    "content": [{"type": "input_text", "text": content}],
                })
            elif isinstance(content, list):
                parts = []
                for item in content:
                    itype = item.get("type") if isinstance(item, dict) else None
                    if itype == "text":
                        parts.append({"type": "input_text", "text": item["text"]})
                    elif itype == "image" and isinstance(item.get("image"), Image.Image):
                        b64 = self._encode_image(item["image"])
                        parts.append({
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{b64}",
                        })
                if parts:
                    result.append({"type": "message", "role": role, "content": parts})
            else:
                result.append({
                    "type": "message",
                    "role": role,
                    "content": [{"type": "input_text", "text": str(content)}],
                })
        return result

    def _anthropic_messages_to_openai(
        self,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Convert the agent's Anthropic-style transcript into OpenAI chat format."""
        result: List[Dict[str, Any]] = []
        if system:
            result.append({"role": "system", "content": system})

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            if role == "assistant" and isinstance(content, list):
                text_parts: List[str] = []
                tool_calls: List[Dict[str, Any]] = []
                for block in content:
                    btype = self._get_block_attr(block, "type")
                    if btype == "text":
                        txt = self._get_block_attr(block, "text", "")
                        if txt:
                            text_parts.append(txt)
                    elif btype == "thinking":
                        pass
                    elif btype == "tool_use":
                        tool_calls.append({
                            "id": self._get_block_attr(block, "id", ""),
                            "type": "function",
                            "function": {
                                "name": self._get_block_attr(block, "name", ""),
                                "arguments": json.dumps(
                                    self._get_block_attr(block, "input", {}) or {}
                                ),
                            },
                        })
                oai_msg: Dict[str, Any] = {
                    "role": "assistant",
                    "content": "\n".join(text_parts) or "",
                }
                if tool_calls:
                    oai_msg["tool_calls"] = tool_calls
                result.append(oai_msg)
            elif role == "user" and isinstance(content, list):
                tool_results = [
                    block for block in content
                    if self._get_block_attr(block, "type") == "tool_result"
                ]
                if tool_results:
                    for tr in tool_results:
                        result.append({
                            "role": "tool",
                            "tool_call_id": self._get_block_attr(tr, "tool_use_id", ""),
                            "content": str(self._get_block_attr(tr, "content", "")),
                        })
                else:
                    text = " ".join(
                        self._get_block_attr(block, "text", str(block))
                        for block in content
                    )
                    result.append({"role": "user", "content": text})
            else:
                result.append({
                    "role": role,
                    "content": str(content) if not isinstance(content, str) else content,
                })
        return result

    def _anthropic_messages_to_responses_input(
        self,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Convert the agent transcript into Responses API input items."""
        result: List[Dict[str, Any]] = []
        msg_counter = 0

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            if role == "assistant" and isinstance(content, list):
                for block in content:
                    btype = self._get_block_attr(block, "type")
                    if btype == "text":
                        txt = self._get_block_attr(block, "text", "")
                        if txt:
                            msg_counter += 1
                            result.append({
                                "type": "message",
                                "id": f"assistant-msg-{msg_counter}",
                                "role": "assistant",
                                "status": "completed",
                                "content": [{
                                    "type": "output_text",
                                    "text": txt,
                                    "annotations": [],
                                }],
                            })
                    elif btype == "tool_use":
                        result.append({
                            "type": "function_call",
                            "call_id": self._get_block_attr(block, "id", ""),
                            "name": self._get_block_attr(block, "name", ""),
                            "arguments": json.dumps(
                                self._get_block_attr(block, "input", {}) or {}
                            ),
                            "status": "completed",
                        })
            elif role == "user" and isinstance(content, list):
                tool_results = [
                    block for block in content
                    if self._get_block_attr(block, "type") == "tool_result"
                ]
                if tool_results:
                    for tr in tool_results:
                        result.append({
                            "type": "function_call_output",
                            "call_id": self._get_block_attr(tr, "tool_use_id", ""),
                            "output": str(self._get_block_attr(tr, "content", "")),
                        })
                else:
                    text = " ".join(
                        self._get_block_attr(block, "text", str(block))
                        for block in content
                    )
                    result.append({
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": text}],
                    })
            else:
                result.append({
                    "type": "message",
                    "role": role,
                    "content": [{
                        "type": "input_text",
                        "text": str(content) if not isinstance(content, str) else content,
                    }],
                })
        return result

    def _parse_structured(self, text: str, text_format: type) -> Any:
        match = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
        if match:
            try:
                return text_format.model_validate_json(match.group())
            except Exception:
                pass
        return text_format.model_validate_json(text)



class ResponsesAPIMixin(OpenAIWireMixin):
    """The OpenAI Responses API call path (`generate` / `create_with_tools`).

    A backend mixing this in supplies `self.client`, `self.model_name`,
    `self.max_tokens`, `self.enable_thinking` and `self.reasoning_effort` in its
    own `__init__`, and whatever bulk-call strategy its endpoint supports.
    """

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1),
           retry=retry_if_not_exception_type(BadRequestError))
    def _stream_create(self, **call_kwargs) -> "_StreamedResp":
        """Streamed responses.create that rebuilds .output / .usage / .output_text.
        Works on real OpenAI and on the LiteLLM :4000 proxy (whose non-streaming
        output[] comes back empty)."""
        items, usage, text = {}, None, ""
        with self.client.responses.create(stream=True, **call_kwargs) as stream:
            for ev in stream:
                et = ev.type
                if et == "response.output_item.done":
                    items[ev.output_index] = ev.item
                elif et == "response.output_text.delta":
                    text += ev.delta or ""
                elif et == "response.completed":
                    usage = getattr(ev.response, "usage", None)
        return _StreamedResp([items[k] for k in sorted(items)], usage, text)

    def generate(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        text_format: Optional[type] = None,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        **kwargs,
    ) -> Any:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = copy.deepcopy(prompt)

        effort = self._resolve_reasoning_effort(
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
        )
        openai_messages = self._to_openai_messages(messages)
        logger.info("[GPTAPI] model=%s reasoning_effort=%s", self.model_name, effort)

        kwargs.pop("temperature", None)
        kwargs.pop("top_p", None)

        try:
            response = self._stream_create(
                model=self.model_name,
                input=openai_messages,
                max_output_tokens=self.max_tokens,
                reasoning={"effort": effort, "summary": "auto"},
                **kwargs,
            )
        except BadRequestError as e:
            logger.error("[GPTAPI] BadRequest for model=%s: %s", self.model_name, e)
            raise
        _u = getattr(response, "usage", None)
        self._last_in_tokens = getattr(_u, "input_tokens", 0) or 0
        self._last_out_tokens = getattr(_u, "output_tokens", 0) or 0
        content = response.output_text or ""

        if text_format is not None:
            return self._parse_structured(content, text_format)
        return content

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1),
           retry=retry_if_not_exception_type(BadRequestError))
    def create_with_tools(
        self,
        *,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        enable_thinking: Optional[bool] = None,
        reasoning_effort: Optional[str] = None,
        **kwargs: Any,
    ) -> _ResponseAdapter:
        """Tool-use / answering path with the Anthropic-style adapter expected by agent.py."""
        effort = self._resolve_reasoning_effort(
            enable_thinking=enable_thinking,
            reasoning_effort=reasoning_effort,
        )

        kwargs.pop("temperature", None)
        kwargs.pop("top_p", None)

        if tools:
            response_input = self._anthropic_messages_to_responses_input(messages)
            call_kwargs = {
                "model": self.model_name,
                "input": response_input,
                "max_output_tokens": kwargs.pop(
                    "max_output_tokens",
                    kwargs.pop("max_completion_tokens", kwargs.pop("max_tokens", self.max_tokens)),
                ),
                "reasoning": {"effort": effort, "summary": "auto"},
                "instructions": system,
                "tools": self._convert_tools_to_responses(tools),
                "tool_choice": "auto",
                "parallel_tool_calls": False,
            }
            call_kwargs.update(kwargs)
            response = self._stream_create(**call_kwargs)

            content_blocks: List[Any] = []
            for item in getattr(response, "output", []) or []:
                itype = getattr(item, "type", None)
                if itype == "reasoning":
                    parts = []
                    for c in getattr(item, "content", None) or []:
                        txt = getattr(c, "text", None)
                        if txt:
                            parts.append(txt)
                    if not parts:
                        for s in getattr(item, "summary", None) or []:
                            txt = getattr(s, "text", None)
                            if txt:
                                parts.append(txt)
                    if parts:
                        content_blocks.append(_ThinkingBlock("\n".join(parts)))
                elif itype == "message":
                    text_parts = []
                    for c in getattr(item, "content", None) or []:
                        txt = getattr(c, "text", None)
                        if txt:
                            text_parts.append(txt)
                    if text_parts:
                        content_blocks.append(_TextBlock("\n".join(text_parts)))
                elif itype == "function_call":
                    try:
                        input_dict = json.loads(getattr(item, "arguments", "") or "{}")
                    except Exception:
                        input_dict = {}
                    content_blocks.append(_ToolUseBlock(
                        getattr(item, "name", ""),
                        input_dict,
                        getattr(item, "call_id", "") or getattr(item, "id", ""),
                    ))

            usage = getattr(response, "usage", None)
            usage_adapter = _UsageAdapter(
                input_tokens=getattr(usage, "input_tokens", 0) or 0,
                output_tokens=getattr(usage, "output_tokens", 0) or 0,
            )
            return _ResponseAdapter(content_blocks, usage_adapter)

        # No-tools path: still use Responses API so reasoning items are returned.
        response_input = self._messages_to_responses_input_simple(messages)
        no_tool_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "input": response_input,
            "max_output_tokens": kwargs.pop(
                "max_output_tokens",
                kwargs.pop("max_completion_tokens", kwargs.pop("max_tokens", self.max_tokens)),
            ),
            "reasoning": {"effort": effort, "summary": "auto"},
            "instructions": system,
        }
        no_tool_kwargs.update(kwargs)
        response = self._stream_create(**no_tool_kwargs)

        content_blocks: List[Any] = []
        for item in getattr(response, "output", []) or []:
            itype = getattr(item, "type", None)
            if itype == "reasoning":
                parts = []
                for c in getattr(item, "content", None) or []:
                    txt = getattr(c, "text", None)
                    if txt:
                        parts.append(txt)
                if not parts:
                    for s in getattr(item, "summary", None) or []:
                        txt = getattr(s, "text", None)
                        if txt:
                            parts.append(txt)
                if parts:
                    content_blocks.append(_ThinkingBlock("\n".join(parts)))
            elif itype == "message":
                text_parts = []
                for c in getattr(item, "content", None) or []:
                    txt = getattr(c, "text", None)
                    if txt:
                        text_parts.append(txt)
                if text_parts:
                    content_blocks.append(_TextBlock("\n".join(text_parts)))

        usage = getattr(response, "usage", None)
        usage_adapter = _UsageAdapter(
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
        )
        return _ResponseAdapter(content_blocks, usage_adapter)



    def generate_batch(self, batch_prompts: List, **kwargs) -> List:
        return [self.generate(p, **kwargs) for p in batch_prompts]

    # ------------------------------------------------------------------
    # Batch API (50% cheaper, asynchronous) — for bulk independent calls
