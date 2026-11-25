from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence

import numpy as np
import pytest

from stylelink.core.database import Database
from stylelink.core.index import VectorIndex
from stylelink.embedding.star import STAR


def make_unit_vectors(n: int, dim: int = 1024, seed: int = 42) -> np.ndarray:
    """Helper to generate `n` deterministic L2-normalised float32 vectors of shape (n, dim)."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim), dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


@pytest.fixture
def mock_embedder(monkeypatch: pytest.MonkeyPatch):
    """Patch STAR so it never loads the 1.3GB model.

    `embed_texts()` hashes each input string into a 1024-D unit vector in float32, while maintaining `progress_callback` and `cancel_check`.
    """

    def fake_init(self) -> None:
        self.device = "cpu"
        self._model = None
        self._tokenizer = None

    def fake_embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        progress_callback: Callable[[int, int], None] | None = None,
        cancel_check: Callable[[], bool] | None = None,
    ):
        if not texts:
            raise ValueError("Texts must not be empty")

        if cancel_check and cancel_check():
            raise InterruptedError("Embedding cancelled")

        total = len(texts)
        if progress_callback:
            progress_callback(0, total)

        vecs = []
        for idx, text in enumerate(texts, start=1):
            if cancel_check and cancel_check():
                raise InterruptedError("Embedding cancelled")

            # Hash text so identical texts get same embedding
            seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little")
            vecs.append(make_unit_vectors(1, dim=1024, seed=seed)[0])

            if progress_callback:
                progress_callback(idx, total)

        return np.vstack(vecs).astype(np.float32)

    monkeypatch.setattr(STAR, "__init__", fake_init)
    monkeypatch.setattr(STAR, "load_components", lambda self: None)
    monkeypatch.setattr(STAR, "release", lambda self: None)
    monkeypatch.setattr(STAR, "embed_texts", fake_embed_texts)


@pytest.fixture
def empty_index() -> VectorIndex:
    """Return an empty VectorIndex."""
    return VectorIndex()


@pytest.fixture
def populated_index() -> tuple[VectorIndex, np.ndarray, np.ndarray]:
    """Return a VectorIndex pre-populated with 5 unit vectors and their IDs.

    Returns:
        (index, vectors, ids) where ids are [10, 20, 30, 40, 50].
    """
    index = VectorIndex()
    vecs = make_unit_vectors(5, dim=1024, seed=123)
    ids = np.array([10, 20, 30, 40, 50], dtype=np.int64)
    index.add(vecs, ids)
    return index, vecs, ids


@pytest.fixture
def memory_db():
    """Yield an in-memory SQLite database and close it after the test finishes."""
    with Database(":memory:") as db:
        yield db
