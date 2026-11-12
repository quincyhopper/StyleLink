import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "logreg.pkl"


class LLRCalibrator:
    """Class for Logistic Regression used for computing log likelihood ratios (LLR)."""

    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH) -> None:
        with open(model_path, "rb") as f:
            self.model: LogisticRegression = pickle.load(f)

    def compute_llr(self, scores: np.ndarray) -> np.ndarray:
        """Compute LLR using trained Logistic Regression model.

        Args:
            scores (np.ndarray): array of similarity scores. Shape (n_queries, k).

        Returns:
            llrs (np.ndarray): array of LLRs. Shape (n_queries, k).

        Raises:
            ValueError: if scores is empty.
        """
        if scores.size == 0:
            raise ValueError("Scores cannot be empty")

        llrs = np.full(scores.shape, np.nan, dtype=np.float32)
        valid = ~np.isnan(scores)  # NaN scores (from VectorIndex.search_by_id) cannot have llrs

        if np.any(valid):
            # Flatten otherwise sklearn will interpret k as features instead of scores
            flat = scores[valid].reshape(-1, 1)
            llrs[valid] = self.model.decision_function(flat) / np.log(10)

        return llrs
