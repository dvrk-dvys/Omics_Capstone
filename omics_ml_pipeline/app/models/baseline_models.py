"""
baseline_models.py — Model registry and hyperopt search spaces.

All models configured for small-n, high-dimensional binary classification
with class imbalance (30 SONFH : 10 control).

Usage (from train_eval_job):
    from app.models.baseline_models import BASELINE_MODELS, HYPEROPT_SPACES, make_pipeline
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from hyperopt import hp
from hyperopt.pyll import scope


# ---------------------------------------------------------------------------
# BASELINE MODELS
# class_weight='balanced' applied wherever sklearn supports it
# ---------------------------------------------------------------------------
BASELINE_MODELS = {
    "logistic_elasticnet": LogisticRegression(
        solver="saga", l1_ratio=0.5,
        C=1.0, max_iter=5000, class_weight="balanced", random_state=42
    ),
    "random_forest": RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42
    ),
    "linear_svc": LinearSVC(
        class_weight="balanced", max_iter=5000, random_state=42
    ),
    "gaussian_nb": GaussianNB(),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "mlp": MLPClassifier(
        hidden_layer_sizes=(64, 32), max_iter=500, random_state=42
    ),
    "xgboost": XGBClassifier(
        n_estimators=200, scale_pos_weight=3,   # 30 SONFH / 10 control
        eval_metric="logloss", random_state=42
    ),
}

# Models that require feature scaling
NEEDS_SCALING = {"logistic_elasticnet", "linear_svc", "knn", "mlp"}


def make_pipeline(name: str, model) -> Pipeline:
    """Wrap model in a StandardScaler pipeline if the model needs scaling."""
    if name in NEEDS_SCALING:
        return Pipeline([("scaler", StandardScaler()), ("model", model)])
    return model


# ---------------------------------------------------------------------------
# HYPEROPT SEARCH SPACES
# Tuned models: xgboost + random_forest
# ---------------------------------------------------------------------------
HYPEROPT_SPACES = {
    "xgboost": {
        "max_depth":        scope.int(hp.quniform("max_depth", 3, 10, 1)),
        "learning_rate":    hp.loguniform("learning_rate", -3, 0),
        "n_estimators":     scope.int(hp.quniform("n_estimators", 100, 400, 50)),
        "subsample":        hp.uniform("subsample", 0.6, 1.0),
        "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 1.0),
        "reg_alpha":        hp.loguniform("reg_alpha", -5, 0),
        "reg_lambda":       hp.loguniform("reg_lambda", -5, 0),
        "scale_pos_weight": 3,
        "eval_metric":      "logloss",
        "random_state":     42,
    },
    "random_forest": {
        "n_estimators":    scope.int(hp.quniform("rf_n_estimators", 100, 400, 50)),
        "max_depth":       scope.int(hp.quniform("rf_max_depth", 3, 20, 1)),
        "min_samples_split": scope.int(hp.quniform("rf_min_samples_split", 2, 10, 1)),
        "max_features":    hp.choice("rf_max_features", ["sqrt", "log2"]),
        "class_weight":    "balanced",
        "random_state":    42,
    },
}
