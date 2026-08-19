"""
Codex GPT backend — GPT-5.x Codex models behind a LiteLLM proxy.

The proxy speaks the OpenAI Responses API and routes the `chatgpt/` model-id
prefix to the Codex backend, so the shared Responses call path applies
unchanged. What is specific to this deployment: its endpoint and key
(`LITELLM_BASE_URL` / `LITELLM_API_KEY` in config.py), the `*-codex` friendly
names, and the absence of an async Batch API — bulk calls run concurrently
in real time instead.

This module does not import `openai_gpt.py`; `models.make_llm` picks between
them by model name.

Example:
    from models.codex_gpt import CodexGPTAPI
    llm = CodexGPTAPI(model_name="gpt-5.4-codex")
    out = llm.generate([{"role": "user", "content": "who are you"}])
"""
from __future__ import annotations

import logging
from typing import List, Optional

from openai import OpenAI

import config
from models._openai_shared import ResponsesAPIMixin

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-5.4-mini-codex"

# Friendly *-codex names -> LiteLLM proxy model ids. A `chatgpt/...` id passed
# straight through is used as-is.
_CODEX_MODEL_MAP = {
    "gpt-5.4-codex":      "chatgpt/gpt-5.4",
    "gpt-5.4-mini-codex": "chatgpt/gpt-5.4-mini",
}
CODEX_NAMES = set(_CODEX_MODEL_MAP)


class CodexGPTAPI(ResponsesAPIMixin):
    """Codex models through the LiteLLM proxy (LITELLM_* in config.py)."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 16384,
        enable_thinking: bool = False,
        reasoning_effort: Optional[str] = None,
    ):
        self.model_name = _CODEX_MODEL_MAP.get(model_name, model_name)
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort if reasoning_effort is not None \
            else (config.LITELLM_REASONING or None)

        key = api_key or config.require("LITELLM_API_KEY")
        url = base_url or config.require("LITELLM_BASE_URL")
        self.client = OpenAI(api_key=key, base_url=url)

    def generate_batch(self, batch_prompts: List, **kwargs) -> List:
        """Concurrent real-time calls via pqdm — the proxy has no async Batch API."""
        from functools import partial
        from pqdm.threads import pqdm as _pqdm
        fn = partial(self.generate, **kwargs)
        return list(_pqdm(batch_prompts, fn, n_jobs=8))

    def generate_batch_api(self, prompts, enable_thinking=None, poll_interval=30,
                           description="batch"):
        """No async Batch API on the proxy — go real-time concurrent."""
        logger.info("[CodexGPTAPI] generate_batch_api → real-time concurrent (%d prompts)",
                    len(prompts))
        return self.generate_batch(prompts, enable_thinking=enable_thinking)

    def __repr__(self) -> str:
        return f"CodexGPTAPI(model={self.model_name!r})"
