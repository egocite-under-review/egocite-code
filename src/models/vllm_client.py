import base64
import copy
import json
import re
from io import BytesIO
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_fixed

import config

BASE_URL = config.VLLM_BASE_URL   # see config.py -> Credentials and endpoints
DEFAULT_MODEL = "Qwen/Qwen3.6-27B-FP8"

# ── Call-mode sampling configs ────────────────────────────────────────────────
#
# Mode 1 — Instruct (no thinking, no preserved thinking)
#   Used when: tool_call=False, enable_thinking=False
_INSTRUCT_PARAMS: Dict[str, Any] = dict(
    temperature=0.7,
    top_p=0.8,
    presence_penalty=1.5,
    extra_body={
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    },
)

# Mode 2 — Thinking, no tool call  (enable_thinking=True, preserve_thinking=False)
#   Used when: tool_call=False, enable_thinking=True
_THINKING_NOTOOL_PARAMS: Dict[str, Any] = dict(
    temperature=0.6,
    top_p=0.95,
    presence_penalty=0.0,
    extra_body={
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": False},
    },
)

# Mode 3 — Thinking with tool call  (enable_thinking=True, preserve_thinking=True)
#   Used when: tool_call=True, enable_thinking=True
_THINKING_TOOL_PARAMS: Dict[str, Any] = dict(
    temperature=0.6,
    top_p=0.95,
    presence_penalty=0.0,
    extra_body={
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
    },
)

# Mode 4 — Tool call without thinking  (enable_thinking=False, preserve_thinking=True)
#   Used when: tool_call=True, enable_thinking=False
_INSTRUCT_TOOL_PARAMS: Dict[str, Any] = dict(
    temperature=0.7,
    top_p=0.8,
    presence_penalty=1.5,
    extra_body={
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False, "preserve_thinking": True},
    },
)

# ── Anthropic-compatible response adapters ────────────────────────────────────
# create_with_tools returns these so agent.py can use the same code path as
# for Claude without knowing which backend produced the response.

class _ThinkingBlock:
    type = "thinking"
    def __init__(self, thinking: str):
        self.thinking = thinking

class _TextBlock:
    type = "text"
    def __init__(self, text: str):
        self.text = text

class _ToolUseBlock:
    type = "tool_use"
    def __init__(self, name: str, input_dict: dict, uid: str):
        self.name = name
        self.input = input_dict
        self.id = uid

class _UsageAdapter:
    cache_read_input_tokens = 0
    cache_creation_input_tokens = 0
    def __init__(self, input_tokens: int, output_tokens: int):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

class _ResponseAdapter:
    def __init__(self, content: list, usage: _UsageAdapter):
        self.content = content
        self.usage = usage


class VLLMClient:
    """Calls a vLLM server (OpenAI-compatible) running at localhost:8001.

    Four call modes (selected automatically by tools + enable_thinking):
      Mode 1 — Instruct              : tools=None,  enable_thinking=False
      Mode 2 — Thinking, no tool     : tools=None,  enable_thinking=True
      Mode 3 — Thinking with tool    : tools=...,   enable_thinking=True
      Mode 4 — Tool call, no thinking: tools=...,   enable_thinking=False

    Both generate() and create_with_tools() return Anthropic-compatible
    objects so agent.py can share the same parsing code for both backends.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        base_url: str = BASE_URL,
        max_tokens: int = 2048,
        enable_thinking: bool = False,
    ):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.client = OpenAI(base_url=base_url, api_key=config.VLLM_API_KEY or "EMPTY")

    # ── internal helpers ──────────────────────────────────────────────────────

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
                        parts.append({"type": "text", "text": item["text"]})
                    elif item.get("type") == "image" and isinstance(item.get("image"), Image.Image):
                        b64 = self._encode_image(item["image"])
                        parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                        })
                result.append({"role": role, "content": parts})
            else:
                result.append({"role": role, "content": str(content)})
        return result

    @staticmethod
    def _convert_tools_to_openai(tools: List[Dict]) -> List[Dict]:
        """Convert Anthropic tool schema → OpenAI function-calling schema."""
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
    def _get_block_attr(block, attr: str, default=None):
        """Works for both object-style (Anthropic SDK) and dict-style blocks."""
        if isinstance(block, dict):
            return block.get(attr, default)
        return getattr(block, attr, default)

    def _anthropic_messages_to_openai(
        self,
        messages: List[Dict],
        system: Optional[str] = None,
    ) -> List[Dict]:
        """Convert Anthropic-style message list to OpenAI format.

        Handles:
        • assistant messages whose content is a list of TextBlock / ThinkingBlock /
          ToolUseBlock objects (as produced by _retrieve_with_tools_once).
        • user messages whose content is a list of tool_result dicts
          ({"type": "tool_result", "tool_use_id": ..., "content": ...}).
        """
        result: List[Dict] = []
        if system:
            result.append({"role": "system", "content": system})

        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")

            if role == "assistant" and isinstance(content, list):
                text_parts: List[str] = []
                tool_calls: List[Dict] = []
                for block in content:
                    btype = self._get_block_attr(block, "type")
                    if btype == "text":
                        t = self._get_block_attr(block, "text", "")
                        if t:
                            text_parts.append(t)
                    elif btype == "thinking":
                        pass  # discard; reasoning is regenerated each call
                    elif btype == "tool_use":
                        tool_calls.append({
                            "id":   self._get_block_attr(block, "id", ""),
                            "type": "function",
                            "function": {
                                "name":      self._get_block_attr(block, "name", ""),
                                "arguments": json.dumps(
                                    self._get_block_attr(block, "input", {}) or {}
                                ),
                            },
                        })
                oai: Dict[str, Any] = {
                    "role": "assistant",
                    "content": "\n".join(text_parts) or "",
                }
                if tool_calls:
                    oai["tool_calls"] = tool_calls
                result.append(oai)

            elif role == "user" and isinstance(content, list):
                tool_results = [
                    b for b in content
                    if self._get_block_attr(b, "type") == "tool_result"
                ]
                if tool_results:
                    for tr in tool_results:
                        result.append({
                            "role":         "tool",
                            "tool_call_id": self._get_block_attr(tr, "tool_use_id", ""),
                            "content":      str(self._get_block_attr(tr, "content", "")),
                        })
                else:
                    text = " ".join(
                        self._get_block_attr(b, "text", str(b))
                        for b in content
                    )
                    result.append({"role": "user", "content": text})

            else:
                result.append({"role": role, "content": str(content) if not isinstance(content, str) else content})

        return result

    @staticmethod
    def _extract_thinking_and_content(msg) -> tuple:
        thinking = (
            getattr(msg, "reasoning_content", None)
            or getattr(msg, "reasoning", None)
            or ""
        )
        content = msg.content or ""
        return thinking, content

    # ── public API ────────────────────────────────────────────────────────────

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
        mode_params = _THINKING_NOTOOL_PARAMS if thinking_flag else _INSTRUCT_PARAMS

        openai_messages = self._to_openai_messages(messages)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=openai_messages,
            max_tokens=self.max_tokens,
            **mode_params,
            **kwargs,
        )
        thinking, content = self._extract_thinking_and_content(response.choices[0].message)
        if thinking:
            content = f"<think>{thinking}</think>\n{content}"

        if text_format is not None:
            return self._parse_structured(content, text_format)
        return content

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def create_with_tools(
        self,
        *,
        messages: List[Dict],
        system: Optional[str] = None,
        tools: Optional[List[Dict]] = None,
        **kw,
    ) -> _ResponseAdapter:
        """Tool-use call that returns an Anthropic-compatible response adapter.

        Mode selection:
          tools + thinking=True    → Mode 3: thinking with tool call
          tools + thinking=False   → Mode 4: tool call without thinking
          tools=None, thinking=True  → Mode 2: thinking, no tool call
          tools=None, thinking=False → Mode 1: instruct
        """
        oai_messages = self._anthropic_messages_to_openai(messages, system=system)
        thinking_flag        = kw.pop("enable_thinking",      self.enable_thinking)
        parallel_tool_calls  = kw.pop("parallel_tool_calls",  True)

        if tools:
            # Mode 3 or 4: tool call — select by enable_thinking
            mode_params = _THINKING_TOOL_PARAMS if thinking_flag else _INSTRUCT_TOOL_PARAMS
            call_kwargs: Dict[str, Any] = dict(
                model=self.model_name,
                messages=oai_messages,
                max_tokens=self.max_tokens,
                **mode_params,
                tools=self._convert_tools_to_openai(tools),
                tool_choice="auto",
                parallel_tool_calls=parallel_tool_calls,
            )
        else:
            # Mode 1 or 2: no tools — select by enable_thinking
            mode_params = _THINKING_NOTOOL_PARAMS if thinking_flag else _INSTRUCT_PARAMS
            call_kwargs = dict(
                model=self.model_name,
                messages=oai_messages,
                max_tokens=self.max_tokens,
                **mode_params,
            )

        response = self.client.chat.completions.create(**call_kwargs)
        msg = response.choices[0].message

        content_blocks: List[Any] = []
        thinking, text = self._extract_thinking_and_content(msg)
        if thinking:
            content_blocks.append(_ThinkingBlock(thinking))
        if text:
            content_blocks.append(_TextBlock(text))
        for tc in msg.tool_calls or []:
            try:
                input_dict = json.loads(tc.function.arguments)
            except Exception:
                input_dict = {}
            content_blocks.append(_ToolUseBlock(tc.function.name, input_dict, tc.id))

        usage = response.usage
        usage_adapter = _UsageAdapter(
            input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage, "completion_tokens", 0) or 0,
        )
        return _ResponseAdapter(content_blocks, usage_adapter)

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

    def __repr__(self) -> str:
        return f"VLLMClient(model={self.model_name!r}, base_url={self.client.base_url!r})"
