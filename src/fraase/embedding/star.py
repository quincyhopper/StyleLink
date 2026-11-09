from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


def detect_best_device() -> str:
    """Detect the best available compute device (CUDA -> MPS -> CPU).

    Returns:
        CUDA, MPS or CPU
    """
    try:
        # To allow FAISS and PyTorch to use OpenMP on the same process
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class STAR:
    """Class for the STAR embedding model."""

    def __init__(self, device: str | None) -> None:
        self.model_path = Path("AIDA-UPM/star")
        self.tokenizer_path = Path("roberta-large")
        self.device = device or detect_best_device()

        self._model = None
        self._tokenizer = None

    def is_cached_locally(self) -> bool:
        """Check whether the STAR model and RoBERTA-large tokenizer exist in the local cache.

        Returns:
            True if both found in cache, False otherwise.
        """
        if self._model is not None and self._tokenizer is not None:
            return True

        # Import function to check the cache, catching import error
        try:
            from huggingface_hub import try_to_load_from_cache
        except ImportError:
            return False

        # Check the model is cached
        cached_bin = try_to_load_from_cache(repo_id=self.model_path, filename="pytorch_model.bin")
        model_cached = isinstance(cached_bin, str)

        # Check the tokenizer is cached
        cached_tokenizer = try_to_load_from_cache(repo_id=self.tokenizer_path, filename="tokenizer.json")
        cached_vocab = try_to_load_from_cache(repo_id=self.tokenizer_path, filename="vocab.json")
        tokenizer_cached = isinstance(cached_tokenizer, str) or isinstance(cached_vocab, str)

        return bool(model_cached and tokenizer_cached)

    def load_components(self) -> None:
        """Load the tokenizer and model on device."""
        if self._model is not None and self._tokenizer is not None:
            return

        # AIDA-UPM/star only has pytorch_model.bin and no .safetensors file. As a result, HuggingFace
        # spawns a background thread (Threat-auto_conversion) to query the Hub, ignoring local_files_only.
        # Prevent this with environment variables:
        os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=self.tokenizer_path, local_files_only=True, device=self.device
        )

        self._model = AutoModel.from_pretrained(
            pretrained_model_name_or_path=self.tokenizer_path, local_files_only=True, device=self.device
        )

        self._model.eval()

    def release(self) -> None:
        """Unload the model and tokenizer and clear device cache."""
        if self._model is None and self._tokenizer is None:
            return

        import gc

        self._model = None
        self._tokenizer = None
        gc.collect()

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass

    def embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        """Generate normalised embeddings for input texts.

        If a document is broken down into > 1 512-token chunks, all chunks are embedded individually and averaged to produce one embedding for the document.

        Args:
            texts: List of text strings.
            batch_size: Batch size for model inference.
            progress_callback: Optional callback receiving (processed_count, total_count).

        Returns:
            2D NumPy array of shape (N, dimension) and dtype float32.

        Raises:
            ValueError: if texts argument is empty.
        """  # ruff:ignore[missing-blank-line-after-summary]
        if not texts:
            raise ValueError("Texts must not be empty.")

        self.load_components()
        import torch

        all_embeddings: list[np.ndarray] = []
        total = len(texts)

        for start_idx in range(0, total, batch_size):
            batch_texts = list(texts[start_idx : start_idx + batch_size])

            # Tokenise into 512-token chunks
            encodings = self._tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_overflowing_tokens=True,
                return_tensors="pt",
            )
            sample_mapping = encodings.pop("overflow_to_sample_mapping").to(self.device)

            total_chunks = len(encodings["input_ids"])
            chunk_outputs: list[torch.Tensor] = []

            # Batch embed the chunks
            for c_start in range(0, total_chunks, batch_size):
                chunk_inputs = {k: v[c_start : c_start + batch_size].to(self.device) for k, v in encodings.items()}
                with torch.no_grad():
                    out = self._model(**chunk_inputs).pooler_output
                    chunk_outputs.append(out)

            # Concat into one large tensor
            all_chunks = (
                torch.cat(chunk_outputs, dim=0) if chunk_outputs else torch.empty((0, 1024), device=self.device)
            )

            # Loop through each original text and find its corresponding embedding(s)
            batch_embeddings: list[torch.Tensor] = []
            for i in range(len(batch_texts)):
                mask = sample_mapping == i  # Elements are True for embeddings matching this text
                chunks_for_current_text = all_chunks[mask]
                if len(chunks_for_current_text) == 1:
                    # Text was only 1 chunk long, so has one corresponding embedding
                    text_embedding = chunks_for_current_text[0]
                else:
                    # Multiple embeddings found: average each dimension across all chunks
                    text_embedding = chunks_for_current_text.mean(dim=0)

                # L2 normalise
                normalised = torch.nn.functional.normalize(text_embedding, p=2, dim=-1)
                batch_embeddings.append(normalised)

            batch_tensor = torch.stack(batch_embeddings)
            all_embeddings.append(batch_tensor.cpu().numpy().astype(np.float32))

            if progress_callback:
                progress_callback(min(start_idx + batch_size, total), total)

        return np.vstack(all_embeddings)
