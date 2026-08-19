"""LLM and embedding backends.

`make_llm(name)` is the entry point every build stage uses: it maps a model-name
string to the right backend. Imports are lazy on purpose — a Qwen-only run
should not need `anthropic` installed, and a memory build should not load
sentence-transformers.

No key or endpoint is hardcoded here: every backend reads its credentials from
the "Credentials and endpoints" section of config.py.
"""

__all__ = ["make_llm", "VLLMClient", "GPTAPI", "CodexGPTAPI", "ClaudeAPI",
           "EmbeddingModel"]


def make_llm(model_name: str):
    """Map a model name to a backend. Routing lives ONLY here — the backend
    modules know nothing about each other, and no two of them claim a name.

    chatgpt/* , *-codex    LiteLLM/Codex proxy  (codex_gpt.CodexGPTAPI)
    gpt / o1 / o3 / o4     api.openai.com       (openai_gpt.GPTAPI)
    *gemma*                vision vLLM server   (vllm_client.VLLMClient)
    anything else          text vLLM server     (vllm_client.VLLMClient)

    Both vLLM branches share one OpenAI-compatible client class and differ only
    in endpoint: Gemma is served by VLLM_VISION_BASE_URL, everything else by
    VLLM_BASE_URL, so the two servers can run side by side.
    """
    import config
    n = model_name.lower()
    if n.startswith("chatgpt/") or n.endswith("-codex"):
        # The proxy streams and has no async Batch API — mark _no_batch so
        # callers fall back to real-time.
        from models.codex_gpt import CodexGPTAPI
        llm = CodexGPTAPI(model_name=model_name)
        llm._no_batch = True
        return llm
    if any(t in n for t in ("gpt", "o1", "o3", "o4")):
        from models.openai_gpt import GPTAPI
        return GPTAPI(model_name=model_name)
    from models.vllm_client import VLLMClient
    if "gemma" in n:
        return VLLMClient(model_name=model_name,
                          base_url=config.VLLM_VISION_BASE_URL)
    return VLLMClient(model_name=model_name)


def __getattr__(name):
    if name == "VLLMClient":
        from models.vllm_client import VLLMClient
        return VLLMClient
    if name == "GPTAPI":
        from models.openai_gpt import GPTAPI
        return GPTAPI
    if name == "CodexGPTAPI":
        from models.codex_gpt import CodexGPTAPI
        return CodexGPTAPI
    if name == "ClaudeAPI":
        from models.claude_api import ClaudeAPI
        return ClaudeAPI
    if name == "EmbeddingModel":
        from models.embedding import EmbeddingModel
        return EmbeddingModel
    raise AttributeError(name)
