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

        # Flatten otherwise sklearn will interpret k as features instead of scores
        flat = scores.reshape(-1, 1)
        llrs = self.model.decision_function(flat) / np.log(10)

        # Reshape back to original shape
        return np.asarray(llrs, dtype=np.float32).reshape(scores.shape)
