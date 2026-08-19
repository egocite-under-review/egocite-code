"""
Wrapper around Anthropic's Claude Messages API (e.g. Claude Sonnet 4.6).

Matches the surface of `model.vllm_client.VLLMClient` (same `.generate(...)`
signature, same `<think>...</think>` framing of reasoning content) so it can
drop into any code that already calls a `VLLMClient`-like instance.

API key comes from `ANTHROPIC_API_KEY` in config.py (or the env var of the
same name).

Differences from the OpenAI-compatible wrappers:
  * Anthropic takes the system prompt as a top-level `system` argument, not as a
    message with role "system" — system messages are pulled out here.
  * Reasoning is requested via `output_config={"effort": "low|medium|high|xhigh|max"}`
    and returned as `thinking` content blocks (re-wrapped in <think>...</think>).
  * Roles must alternate; consecutive same-role messages are merged here so the
    agent's retry/feedback turns can't trip Anthropic's alternation check.

Example usage:
    from models.claude_api import ClaudeAPI
    llm = ClaudeAPI(model_name="claude-sonnet-4-6", enable_thinking=True)
    out = llm.generate([{"role": "user", "content": "who are you"}])
"""
from __future__ import annotations

import base64
import copy
import re
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Union

import anthropic
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_fixed

import config

DEFAULT_MODEL = "claude-sonnet-4-6"

# Valid reasoning effort levels accepted by output_config.
_VALID_EFFORTS = ("low", "medium", "high", "xhigh", "max")


class ClaudeAPI:
    """Drop-in for VLLMClient that targets Anthropic's Claude Messages API."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        max_tokens: int = 8192,
        enable_thinking: bool = False,
        effort: str = "medium",
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        if effort not in _VALID_EFFORTS:
            raise ValueError(
                f"ClaudeAPI effort must be one of {_VALID_EFFORTS}, got {effort!r}"
            )
        self.effort = effort

        key = api_key or config.require("ANTHROPIC_API_KEY")
        self.client = anthropic.Anthropic(api_key=key)
        self.kwargs = {}  # compatibility with WorldMM LLMModel.__repr__

    # ------------------------------------------------------------------
    # Message → Anthropic format
    # ------------------------------------------------------------------
    def _encode_image(self, image: Image.Image) -> str:
        buf = BytesIO()
        image.save(buf, format="JPEG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def _content_to_blocks(self, content: Any) -> Any:
        """Convert one message's content to Anthropic content (str or block list)."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if item.get("type") == "text":
                    parts.append({"type": "text", "text": item["text"]})
                elif item.get("type") == "image" and isinstance(item.get("image"), Image.Image):
                    parts.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": self._encode_image(item["image"]),
                        },
                    })
            return parts
        return str(content)

    def _to_anthropic_messages(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """Split out system prompt(s); merge consecutive same-role turns."""
        system_parts: List[str] = []
        out: List[Dict[str, Any]] = []
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            if role == "system":
                # Anthropic system is a single string; collect text only.
                if isinstance(content, str):
                    system_parts.append(content)
                elif isinstance(content, list):
                    system_parts.extend(
                        i.get("text", "") for i in content if i.get("type") == "text"
                    )
                continue
            blocks = self._content_to_blocks(content)
            # Merge into the previous turn if same role (keeps strict alternation).
            if out and out[-1]["role"] == role:
                prev = out[-1]["content"]
                if isinstance(prev, str) and isinstance(blocks, str):
                    out[-1]["content"] = prev + "\n" + blocks
                else:
                    prev_list = prev if isinstance(prev, list) else [{"type": "text", "text": prev}]
                    new_list = blocks if isinstance(blocks, list) else [{"type": "text", "text": blocks}]
                    out[-1]["content"] = prev_list + new_list
            else:
                out.append({"role": role, "content": blocks})
        system = "\n\n".join(p for p in system_parts if p) or None
        return system, out

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def generate(
        self,
        prompt: Union[str, List[Dict[str, Any]]],
        text_format: Optional[type] = None,
        enable_thinking: Optional[bool] = None,
        **kwargs,
    ) -> Any:
        if isinstance(prompt, str):
            messages = [{"role": "user", "content": prompt}]
        else:
            messages = copy.deepcopy(prompt)

        thinking_flag = self.enable_thinking if enable_thinking is None else enable_thinking
        system, anthropic_messages = self._to_anthropic_messages(messages)
        if not anthropic_messages:
            # Anthropic requires at least one (user) message besides `system`.
            raise ValueError("ClaudeAPI.generate: no user/assistant messages to send "
                             "(only a system prompt was provided).")

        create_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "messages": anthropic_messages,
        }
        if system:
            create_kwargs["system"] = system

        if thinking_flag:
            create_kwargs["output_config"] = {"effort": self.effort}
            # Extended thinking forbids a custom temperature; drop it if passed.
            kwargs.pop("temperature", None)

        create_kwargs.update(kwargs)
        response = self.client.messages.create(**create_kwargs)
        _u = getattr(response, "usage", None)
        self._last_in_tokens = getattr(_u, "input_tokens", 0) or 0
        self._last_out_tokens = getattr(_u, "output_tokens", 0) or 0

        thinking_text, answer_text = "", ""
        for block in response.content:
            btype = getattr(block, "type", None)
            if btype == "thinking":
                thinking_text += getattr(block, "thinking", "") or ""
            elif btype == "text":
                answer_text += getattr(block, "text", "") or ""

        content = answer_text
        if thinking_text:
            content = f"<think>{thinking_text}</think>\n{answer_text}"

        if text_format is not None:
            return self._parse_structured(content, text_format)
        return content

    def _parse_structured(self, text: str, text_format: type) -> Any:
        match = re.search(r"\{.*\}|\[.*\]", text, re.DOTALL)
        if match:
            try:
                return text_format.model_validate_json(match.group())
            except Exception:
                pass
        return text_format.model_validate_json(text)

    def generate_batch(self, batch_prompts: List, **kwargs) -> List:
        return [self.generate(p, **kwargs) for p in batch_prompts]

    # ------------------------------------------------------------------
    # Low-level tool-use API (for the interleaved-thinking agent loop)
    # ------------------------------------------------------------------
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def create_with_tools(
        self,
        *,
        messages: List[Dict[str, Any]],
        system: Optional[str] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        enable_thinking: Optional[bool] = None,
        interleaved_thinking: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Raw Anthropic ``messages.create`` call with tools + interleaved thinking.

        Returns the full ``Message`` response object so the caller can walk its
        ``content`` blocks (``thinking`` / ``text`` / ``tool_use``) and append
        the corresponding tool_result blocks back into ``messages``.
        Messages MUST be in Anthropic format already (no role:system inline —
        pass that via ``system``).
        """
        thinking_flag = self.enable_thinking if enable_thinking is None else enable_thinking
        kwargs.pop("parallel_tool_calls", None)  # Anthropic API does not support this

        # Convert any PIL Image items in user message content to Anthropic base64 format.
        converted: List[Dict[str, Any]] = []
        for msg in messages:
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                new_content = []
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "image" and isinstance(item.get("image"), Image.Image):
                        new_content.append({
                            "type": "image",
                            "source": {"type": "base64", "media_type": "image/jpeg", "data": self._encode_image(item["image"])},
                        })
                    else:
                        new_content.append(item)
                converted.append({**msg, "content": new_content})
            else:
                converted.append(msg)

        create_kwargs: Dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "messages": converted,
        }
        if system:
            create_kwargs["system"] = system
        if tools:
            create_kwargs["tools"] = tools
            if "tool_choice" not in kwargs:
                create_kwargs["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
        if thinking_flag:
            create_kwargs["output_config"] = {"effort": self.effort}
            # Extended thinking forbids a custom temperature.
            kwargs.pop("temperature", None)
        create_kwargs.update(kwargs)

        extra_headers: Dict[str, str] = {}
        if interleaved_thinking and thinking_flag:
            # Beta header for interleaving thinking with tool use.
            extra_headers["anthropic-beta"] = "interleaved-thinking-2025-05-14"

        return self.client.messages.create(extra_headers=extra_headers, **create_kwargs)

    def __repr__(self) -> str:
        return f"ClaudeAPI(model={self.model_name!r})"
