"""
OpenAI GPT backend — models served by api.openai.com.

Matches the surface of `models.vllm_client.VLLMClient` (same `.generate(...)`
signature) so it drops into any code that calls a `VLLMClient`-like instance.

This deployment owns `OPENAI_API_KEY` / `OPENAI_BASE_URL` in config.py, and is
the only one with a real async Batch API. The Responses API call path itself is
shared (models/_openai_shared.py). This module does not import `codex_gpt.py`;
`models.make_llm` picks between them by model name.

GPT-5 reasoning models:
  * use `max_completion_tokens` (not `max_tokens`);
  * expose a `reasoning_effort` knob instead of an explicit thinking toggle, and
    do NOT return raw chain-of-thought — so output has no <think> block;
  * only support the default temperature, so we never set it.
`enable_thinking` maps to reasoning_effort: True -> high, False -> none.

Example:
    from models.openai_gpt import GPTAPI
    llm = GPTAPI(model_name="gpt-5.5", enable_thinking=True)
    out = llm.generate([{"role": "user", "content": "who are you"}])
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Union

from openai import OpenAI

import config
from models._openai_shared import (
    ResponsesAPIMixin, _batch_error_msg, _EFFORT_THINKING, _EFFORT_NON_THINKING,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-5.4-mini-2026-03-17"

_MODEL_MAP: dict = {}


class GPTAPI(ResponsesAPIMixin):
    """Drop-in for VLLMClient that targets OpenAI's GPT (e.g. GPT-5.5)."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_tokens: int = 16384,
        enable_thinking: bool = False,
        reasoning_effort: Optional[str] = None,
    ):
        self.model_name = _MODEL_MAP.get(model_name, model_name)
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.reasoning_effort = reasoning_effort  # override the enable_thinking mapping

        key = api_key or config.require("OPENAI_API_KEY")
        url = base_url or config.OPENAI_BASE_URL
        self.client = OpenAI(api_key=key, base_url=url) if url else OpenAI(api_key=key)

    # ------------------------------------------------------------------
    # Batch API (50% cheaper, asynchronous) — for bulk independent calls
    # ------------------------------------------------------------------
    def generate_batch_api(
        self,
        prompts: List[Union[str, List[Dict[str, Any]]]],
        enable_thinking: Optional[bool] = None,
        poll_interval: int = 30,
        description: str = "batch",
    ) -> List[str]:
        """Run many independent prompts through the OpenAI Batch API (50% cheaper).

        Submits all prompts as ONE async batch, polls until it completes, then
        returns the raw text outputs aligned to the input order. A per-request
        failure yields "" for that slot. Blocks until done (SLA up to 24h; small
        jobs usually finish in minutes).
        """
        if not prompts:
            return []
        thinking_flag = self.enable_thinking if enable_thinking is None else enable_thinking
        effort = self.reasoning_effort or (
            _EFFORT_THINKING if thinking_flag else _EFFORT_NON_THINKING)

        # 1. Build the JSONL request file (Batch API only supports /v1/chat/completions).
        lines = []
        for i, prompt in enumerate(prompts):
            msgs = [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
            body = {
                "model": self.model_name,
                "messages": self._to_openai_messages(msgs),
                "max_completion_tokens": self.max_tokens,
            }
            if effort != "none":
                body["reasoning_effort"] = effort
            lines.append(json.dumps({
                "custom_id": f"req-{i}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }))
        jsonl = ("\n".join(lines)).encode("utf-8")

        # 2. Upload + create the batch.
        extra_headers = getattr(self, "_batch_extra_headers", None) or {}
        up = self.client.files.create(
            file=("batch_input.jsonl", jsonl),
            purpose="batch",
            extra_headers=extra_headers,
        )
        batch = self.client.batches.create(
            input_file_id=up.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": description},
            extra_headers=extra_headers,
        )
        logger.info("[Batch] model=%s reasoning_effort=%s — submitting %d requests (batch=%s) polling every %ds",
                    self.model_name, effort, len(prompts), batch.id, poll_interval)

        # 3. Poll to completion.
        while batch.status not in ("completed", "failed", "expired", "cancelled"):
            time.sleep(poll_interval)
            batch = self.client.batches.retrieve(batch.id)
            counts = getattr(batch, "request_counts", None)
            logger.info("[Batch] %s %s status=%s (%s)", description, batch.id, batch.status,
                        f"{getattr(counts,'completed',0)}/{getattr(counts,'total',len(prompts))}"
                        if counts else "")
        if batch.status != "completed":
            errs = getattr(getattr(batch, "errors", None), "data", None) or []
            detail = "; ".join(
                dict.fromkeys(getattr(e, "message", str(e)) for e in errs)) if errs else ""
            raise RuntimeError(
                f"Batch {batch.id} ended with status={batch.status}"
                + (f": {detail}" if detail else ""))

        results: List[str] = [""] * len(prompts)
        n_err = 0

        # Successful requests (output_file_id is None only when ALL requests failed).
        out_id = getattr(batch, "output_file_id", None)
        if out_id:
            for line in self.client.files.content(out_id).text.splitlines():
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                cid = obj.get("custom_id", "?")
                try:
                    idx = int(cid.split("-")[1])
                except Exception:
                    idx = None
                resp = obj.get("response") or {}
                if resp.get("status_code") == 200:
                    if idx is not None:
                        try:
                            results[idx] = resp["body"]["choices"][0]["message"]["content"] or ""
                            _u = resp["body"].get("usage") or {}
                            self._batch_in_tokens  = getattr(self, "_batch_in_tokens",  0) + (_u.get("prompt_tokens",     0) or 0)
                            self._batch_out_tokens = getattr(self, "_batch_out_tokens", 0) + (_u.get("completion_tokens", 0) or 0)
                        except Exception:
                            results[idx] = ""
                else:
                    # Non-200 in the output file — log it so it's not silently dropped.
                    n_err += 1
                    logger.warning("[Batch error] %s %s: %s",
                                   description, cid, _batch_error_msg(obj))

        # Failed requests land in the error file — log EVERY one to stdout/console.
        err_id = getattr(batch, "error_file_id", None)
        first_err = ""
        if err_id:
            err_lines = [l for l in self.client.files.content(err_id).text.splitlines() if l.strip()]
            n_err += len(err_lines)
            for el in err_lines:
                try:
                    obj = json.loads(el)
                    cid, msg = obj.get("custom_id", "?"), _batch_error_msg(obj)
                except Exception:
                    cid, msg = "?", el[:500]
                if not first_err:
                    first_err = f"{cid}: {msg}"
                logger.warning("[Batch error] %s %s: %s", description, cid, msg)

        if not out_id:
            raise RuntimeError(
                f"Batch {batch.id} produced no output (all {len(prompts)} requests failed). "
                f"First error: {first_err or '(no error file returned)'}"
            )

        n_ok = sum(1 for r in results if r)
        logger.info("[Batch] %s %s completed — %d/%d returned content (%d errored)",
                    description, batch.id, n_ok, len(prompts), n_err)
        return results

    def __repr__(self) -> str:
        return f"GPTAPI(model={self.model_name!r})"
