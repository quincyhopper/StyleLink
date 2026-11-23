import json
from pathlib import Path

import numpy as np

CALIBRATORS_PATH = Path(__file__).resolve().parent / "calibrators.json"
with open(CALIBRATORS_PATH, encoding="utf-8") as f:
    CALIBRATORS: dict[str, list[float]] = json.load(f)


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

        self.coef, self.intercept = CALIBRATORS[name]

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

        return ((self.coef * scores + self.intercept) / np.log(10)).astype(np.float32)


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

    hypothesis = "Same Person" if llr > 0 else "Different Person"
    return f"{strength} support for {hypothesis}"
