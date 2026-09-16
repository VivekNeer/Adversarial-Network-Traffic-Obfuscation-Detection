"""Classical baselines on the flow-statistics view.

Three of them, chosen because each answers a different objection to the deep models.

* **Random forest** -- the strongest thing you can do with these features without
  a neural network, and the standard baseline in the intrusion-detection
  literature. It also reports feature importances, which is how the project can
  say *which* statistics carry the obfuscation signal rather than only reporting
  that a network found it.
* **RBF SVM** -- a genuinely different inductive bias (margin over a kernel
  similarity, not axis-aligned splits), so agreement between it and the forest is
  evidence about the features rather than about one learner.
* **Logistic regression** -- a linear decision boundary. Its gap to the others is
  a direct measurement of how much of this problem is non-linear, which is the
  quantitative case for using a deep model at all.

Everything here is fitted on **scaled** statistics using the scaler fitted on the
training split, so the baselines and the neural models see exactly the same inputs
and a comparison between them is about the learner, not the preprocessing.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from antod.data.features import FEATURE_NAMES


class SklearnBaseline:
    """Thin wrapper giving the sklearn estimators the interface the evaluator wants."""

    def __init__(self, name: str, estimator) -> None:
        self.name = name
        self.estimator = estimator

    def fit(self, stats: np.ndarray, y: np.ndarray) -> SklearnBaseline:
        self.estimator.fit(stats, y)
        return self

    def predict(self, stats: np.ndarray) -> np.ndarray:
        return self.estimator.predict(stats)

    def predict_proba(self, stats: np.ndarray) -> np.ndarray:
        return self.estimator.predict_proba(stats)

    def feature_importance(self) -> dict[str, float] | None:
        """Per-feature importance where the estimator exposes one.

        Tree ensembles give impurity-based importances directly. For a linear
        model the analogue is the magnitude of the coefficient, summed over the
        one-vs-rest rows -- valid here only because every feature was standardised
        to comparable scale first.
        """
        est = self.estimator
        if hasattr(est, "feature_importances_"):
            values = np.asarray(est.feature_importances_, dtype=np.float64)
        elif hasattr(est, "coef_"):
            values = np.abs(np.asarray(est.coef_, dtype=np.float64)).sum(axis=0)
            values = values / max(values.sum(), 1e-12)
        else:
            return None
        return dict(zip(FEATURE_NAMES, values.tolist(), strict=True))

    def top_features(self, k: int = 15) -> list[tuple[str, float]]:
        imp = self.feature_importance()
        if imp is None:
            return []
        return sorted(imp.items(), key=lambda kv: kv[1], reverse=True)[:k]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"name": self.name, "estimator": self.estimator}, path)

    @classmethod
    def load(cls, path: str | Path) -> SklearnBaseline:
        blob = joblib.load(Path(path))
        return cls(name=blob["name"], estimator=blob["estimator"])


def random_forest(seed: int = 42, **kwargs) -> SklearnBaseline:
    params = {
        "n_estimators": 400,
        "max_depth": None,
        "min_samples_leaf": 2,
        "n_jobs": -1,
        "class_weight": "balanced",
        "random_state": seed,
    }
    params.update(kwargs)
    return SklearnBaseline("random_forest", RandomForestClassifier(**params))


def rbf_svm(seed: int = 42, cv: int = 3, **kwargs) -> SklearnBaseline:
    """RBF SVM wrapped in probability calibration.

    An SVM has no native notion of class probability, and the ROC/AUC figures in
    §5 need one. ``SVC(probability=True)`` is deprecated in current scikit-learn,
    so calibration is done explicitly: ``ensemble=False`` fits a single calibrator
    on cross-validated predictions rather than averaging ``cv`` separate SVMs,
    which keeps the decision function the one we actually trained.
    """
    params = {
        "C": 10.0,
        "gamma": "scale",
        "kernel": "rbf",
        "class_weight": "balanced",
        "random_state": seed,
    }
    params.update(kwargs)
    return SklearnBaseline(
        "rbf_svm", CalibratedClassifierCV(SVC(**params), method="sigmoid", cv=cv, ensemble=False)
    )


def logistic_regression(seed: int = 42, **kwargs) -> SklearnBaseline:
    params = {
        "max_iter": 2000,
        "C": 1.0,
        "class_weight": "balanced",
        "random_state": seed,
    }
    params.update(kwargs)
    return SklearnBaseline("logistic_regression", LogisticRegression(**params))


BASELINES: dict[str, Callable[..., SklearnBaseline]] = {
    "random_forest": random_forest,
    "rbf_svm": rbf_svm,
    "logistic_regression": logistic_regression,
}


def build_baseline(name: str, seed: int = 42, **kwargs) -> SklearnBaseline:
    if name not in BASELINES:
        raise KeyError(f"unknown baseline {name!r}; available: {sorted(BASELINES)}")
    return BASELINES[name](seed=seed, **kwargs)
