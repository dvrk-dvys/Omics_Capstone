"""
univariate_models.py — Model registry for the univariate single-probe wrapper.

Non-ANN models for use inside the univariate wrapper loop.
professor_ann is handled separately inside univariate_ann.py because it uses
the custom faithful PyTorch implementation.

Usage (from univariate_ann.py):
    from app.models.univariate_models import get_univariate_models, UNIVARIATE_NEEDS_SCALING
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def get_univariate_models(random_state: int = 42) -> dict:
    """
    Return registry of non-ANN univariate single-probe models.
    Each call returns fresh (unfitted) model instances.

    class_weight='balanced' applied to both — mirrors the imbalance setting
    used in baseline_models.py (30 SONFH : 10 control).
    """
    return {
        "logistic": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        ),
        "svm_rbf": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=random_state,
        ),
    }


# Module-level default registry (random_state=42 — overridden at call time in practice)
UNIVARIATE_MODELS = get_univariate_models()

# Models that expect scaled input (z-score already applied upstream in run_wrapper)
UNIVARIATE_NEEDS_SCALING = {"logistic", "svm_rbf"}
