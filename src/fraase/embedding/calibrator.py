import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

CALIBRATOR_DIR = Path(__file__).resolve().parent

CALIBRATORS = {
    "General": CALIBRATOR_DIR / "general_calibrator.pkl",
    "Forum posts": CALIBRATOR_DIR / "forum_calibrator.pkl",
    "Chat logs": CALIBRATOR_DIR / "chat_calibrator.pkl",
}


class LLRCalibrator:
    """Class for Logistic Regression used for computing log likelihood ratios (LLR)."""

    def __init__(self, name: str = "General") -> None:
        """Initialise the calibrator model used for turning cosine similarity scores into LLRs.

        Args:
           name (str): the name of the calibration model to use.

        Raises:
            ValueError: if uknown name parameter is passed.
        """
        if name not in CALIBRATORS:
            raise ValueError(f"Uknown calibrator: {name}. Choose from {list(CALIBRATORS)}")
        with open(CALIBRATORS[name], "rb") as f:
            self.model: LogisticRegression = pickle.load(f)

    def compute_llr(self, scores: np.ndarray) -> np.ndarray:
        """Compute LLR using trained Logistic Regression model.

        Args:
            scores (np.ndarray): array of similarity scores. Shape (n_queries, k).

        Returns:
            llrs (np.ndarray): array of LLRs in base10. Shape (n_queries, k).

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
            llrs[valid] = self.model.decision_function(flat) / np.log(10)  # base10

        return llrs


def llr_to_verbal(llr: float) -> str:
    """Convert a base10 LLR to a verbal interpretation.

    Args:
        llr (float): a log-likelihood ratio in base10.

    Returns:
        A string describing the degree of support for each hypothesis.

    Raises:
        TypeError: if llr is None.
    """
    if llr is None:
        raise TypeError("Invalid llr.")

    mag = abs(llr)
    if mag < 0.1:
        return "Evidence is inconclusive."

    if mag > 4:
        strength = "Extremely strong"
    elif mag >= 3:
        strength = "Very strong"
    elif mag >= 2:
        strength = "Strong"
    elif mag >= 1:
        strength = "Moderate"
    elif mag >= 0.5:
        strength = "Weak"
    else:
        strength = "Very weak"

    hypothesis = "Same Person" if llr > 0 else "Different People"
    return f"{strength} support for {hypothesis}"
