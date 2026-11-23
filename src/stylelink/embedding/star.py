from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

MODEL_DIR = Path(__file__).resolve().parent / "star_model"


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

    def __init__(self) -> None:
        self.device = detect_best_device()

        self._model = None
        self._tokenizer = None

    def load_components(self) -> None:
        """Load the tokenizer and model on device."""
        if self._model is not None and self._tokenizer is not None:
            return

        from transformers import AutoModel, AutoTokenizer

        # The packaged app bundles already include MODEL_DIR. For git clones, star_model is ignored because it's ~1.3GB,
        # so we download and save it on the first run.
        if not MODEL_DIR.exists():
            # To avoid leaving a duplicate copy in ~/.cache/hugginface/hub/, cache in a temporary dir that is eventually deteted.
            import tempfile

            # AIDA-UPM/star only has pytorch_model.bin and no .safetensors file. As a result, HuggingFace
            # spawns a background thread (Threat-auto_conversion) that tries to download a second model.safetensors
            # copy into tmp. When the `with` block finishes and deletes `tmp`, the background thread is still mid-download
            # so we must disable this safetensors conversion
            os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")
            with tempfile.TemporaryDirectory() as tmp:
                AutoTokenizer.from_pretrained("roberta-large", cache_dir=tmp).save_pretrained(MODEL_DIR)
                AutoModel.from_pretrained("AIDA-UPM/star", cache_dir=tmp).save_pretrained(MODEL_DIR)

        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True)
        self._model = AutoModel.from_pretrained(MODEL_DIR, local_files_only=True).to(self.device)
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
        cancel_check: Callable[[], bool] | None = None,
    ) -> np.ndarray:
        """Generate normalised embeddings for input texts.

        If a document is broken down into > 1 512-token chunks, all chunks are embedded individually and averaged to produce one embedding for the document.

        Args:
            texts: List of text strings.
            batch_size: Batch size for model inference.
            progress_callback: Optional callback receiving (processed_count, total_count).
            cancel_check: If True, the user has requested to cancel the embedding.

        Returns:
            2D NumPy array of shape (N, dimension) and dtype float32.

        Raises:
            ValueError: if texts argument is empty.
            InterruptedError: if embedding is cancelled.
        """
        if not texts:
            raise ValueError("Texts must not be empty.")

        self.load_components()
        if cancel_check and cancel_check():
            # If user cancels during model download, the model will download but the actual embedding will be stopped
            raise InterruptedError("Embedding cancelled")

        import torch

        all_embeddings: list[np.ndarray] = []
        total = len(texts)
        if progress_callback:
            progress_callback(0, total)

        for start_idx in range(0, total, batch_size):
            if cancel_check and cancel_check():
                raise InterruptedError("Embedding cancelled.")
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
