"""Training-side rank filtering and frozen numerical transformations."""
import numpy as np
from scipy.linalg import qr
from sklearn.preprocessing import QuantileTransformer, StandardScaler


class FeatureTransform:
    def __init__(self, mode, *, quantiles=1000, seed=42):
        if mode not in ("quantile", "standard"):
            raise ValueError("mode must be quantile or standard")
        self.mode, self.quantiles, self.seed = mode, quantiles, seed

    def fit(self, values, names):
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(names) or len(values) < 2:
            raise ValueError("Expected at least two training rows and named feature columns")
        if not np.isfinite(values).all():
            raise ValueError("Preprocessing requires finite numerical features")
        self.names_ = list(names)
        centred = values - values.mean(axis=0)
        scales = np.sqrt(np.mean(centred**2, axis=0))
        nonconstant = np.flatnonzero(scales > 0)
        if not len(nonconstant):
            raise ValueError("No nonconstant features remain in the training data")
        normalized = centred[:, nonconstant] / scales[nonconstant]
        _, upper, permutation = qr(normalized, mode="economic", pivoting=True)
        diagonal = np.abs(np.diag(upper))
        tolerance = np.finfo(float).eps * max(normalized.shape) * diagonal.max()
        rank = int(np.sum(diagonal > tolerance))
        self.indices_ = np.sort(nonconstant[permutation[:rank]])
        self.retained_names_ = [self.names_[i] for i in self.indices_]
        self.dropped_names_ = [name for i, name in enumerate(self.names_) if i not in self.indices_]
        self.transformer_ = (StandardScaler() if self.mode == "standard" else QuantileTransformer(
            n_quantiles=min(self.quantiles, len(values)), output_distribution="normal",
            subsample=None, random_state=self.seed))
        self.transformer_.fit(values[:, self.indices_])
        self.training_rows_ = len(values)
        return self

    def transform(self, values):
        values = np.asarray(values, dtype=float)
        if values.ndim != 2 or values.shape[1] != len(self.names_) or not np.isfinite(values).all():
            raise ValueError("New feature values must match the original finite input schema")
        return self.transformer_.transform(values[:, self.indices_])

    def audit(self):
        return {"mode": self.mode, "training_rows": self.training_rows_,
                "retained_features": self.retained_names_, "dropped_features": self.dropped_names_}


class TargetTransform:
    def fit(self, values):
        values = np.asarray(values, dtype=float)
        if not np.isfinite(values).all() or not values.size:
            raise ValueError("Training targets must be finite and nonempty")
        self.mean_ = float(values.mean())
        self.scale_ = float(values.std()) or 1.0
        return self

    def transform(self, values):
        return (np.asarray(values) - self.mean_) / self.scale_

    def inverse_transform(self, values):
        return np.asarray(values) * self.scale_ + self.mean_
