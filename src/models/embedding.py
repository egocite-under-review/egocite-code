from typing import List, Union

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Local Qwen3 text embedding via SentenceTransformers."""

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B", text_model_name: str = None, device: str = "auto"):
        model_name = text_model_name or model_name
        self.model_name = model_name
        self.model = SentenceTransformer(
            model_name,
            model_kwargs={"attn_implementation": "flash_attention_2", "dtype": "auto", "device_map": device},
            tokenizer_kwargs={"padding_side": "left"},
        )

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 256,
        is_query: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Encode texts. For asymmetric retrieval models (e.g. Qwen3-Embedding),
        set is_query=True when encoding a search query so the model applies
        the query-side instruction prefix.
        """
        if isinstance(texts, str):
            texts = [texts]
        if is_query:
            # SentenceTransformer-style: use the preset "query" prompt if the
            # model defines one (Qwen3-Embedding models do). Falls back to
            # raw text for models without query prompts.
            try:
                return self.model.encode(texts, batch_size=batch_size, prompt_name="query")
            except (KeyError, ValueError):
                pass
        return self.model.encode(texts, batch_size=batch_size)

    def __repr__(self) -> str:
        return f"EmbeddingModel(model={self.model_name!r})"
