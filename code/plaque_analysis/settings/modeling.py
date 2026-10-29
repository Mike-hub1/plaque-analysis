"""Configuration for the Figure 5 methods implementation."""
from dataclasses import dataclass, field
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class NetworkConfig:
    dimension: int = 192
    layers: int = 3
    heads: int = 8
    feedforward_ratio: float = 4 / 3
    attention_dropout: float = 0.2
    feedforward_dropout: float = 0.1
    residual_dropout: float = 0.0
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    batch_size: int = 128
    max_epochs: int = 300
    patience: int = 16
    device: str = "cpu"

    def validate(self):
        if type(self.heads) is not int or self.heads < 1 or type(self.dimension) is not int or self.dimension < 2 or self.dimension % self.heads:
            raise ValueError("Network dimension must be divisible by positive attention heads")
        if any(type(value) is not int or value < 1 for value in (self.layers, self.batch_size, self.max_epochs, self.patience)):
            raise ValueError("Layers, batch size, epochs and patience must be positive")
        if not all(math.isfinite(value) for value in (self.feedforward_ratio, self.learning_rate, self.weight_decay)) or self.feedforward_ratio <= 0 or int(self.dimension * self.feedforward_ratio) < 1 or self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("Invalid network optimisation parameters")
        for value in (self.attention_dropout, self.feedforward_dropout, self.residual_dropout):
            if not 0 <= value < 1:
                raise ValueError("Dropout must be in [0, 1)")


@dataclass(frozen=True)
class Config:
    id_column: str = "plaque_id"
    animal_column: str = "animal_id"
    age_column: str = "age_months"
    target_column: str = "lamp1_area_um2"
    baseline: tuple = ("abeta_area_um2", "age_months")
    feature_blocks: dict = field(default_factory=lambda: {
        "F": [f"F{i:02d}" for i in range(1, 41)],
        "Popt": [f"Popt{i:02d}" for i in range(1, 16)],
        "PPA": [f"PPA{i:02d}" for i in range(1, 6)]})
    construction_animals: tuple = ()
    inner_folds: int = 7
    quantiles: int = 1000
    seed: int = 42
    bootstrap_repeats: int = 10000
    stability_repeats: int = 100
    animals_per_age: int = 2
    k_candidates: tuple = tuple(range(2, 9))
    gmm_initializations: int = 10
    gmm_max_iterations: int = 500
    covariance_regularization: float = 1e-6
    save_network_weights: bool = False
    network: NetworkConfig = field(default_factory=NetworkConfig)

    def validate(self):
        self.network.validate()
        if self.inner_folds < 2 or self.quantiles < 2:
            raise ValueError("At least two inner folds and quantiles are required")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32 or any(type(value) is not int or value < 1 for value in (self.bootstrap_repeats, self.stability_repeats, self.animals_per_age)):
            raise ValueError("Seed must be nonnegative; repeated evaluation counts must be positive")
        if not self.k_candidates or any(type(value) is not int or value < 2 for value in self.k_candidates):
            raise ValueError("GMM candidate class counts must be at least two")
        if len(set(self.k_candidates)) != len(self.k_candidates):
            raise ValueError("GMM candidate class counts must be unique")
        if self.gmm_initializations < 1 or self.gmm_max_iterations < 1 or self.covariance_regularization <= 0:
            raise ValueError("Invalid GMM settings")
        if set(self.feature_blocks) != {"F", "Popt", "PPA"}:
            raise ValueError("feature_blocks must contain exactly F, Popt and PPA")
        if any(not columns for columns in self.feature_blocks.values()):
            raise ValueError("Each imaging block must have at least one declared feature")
        imaging = self.imaging_features()
        names = list(self.baseline) + imaging
        if len(names) != len(set(names)):
            raise ValueError("Baseline and imaging feature roles must be disjoint and unique")
        if len(self.baseline) != 2 or self.age_column not in self.baseline:
            raise ValueError("B must contain Aβ area and the declared age column")
        if self.target_column in names or self.id_column in names or self.animal_column in names:
            raise ValueError("Targets and identifiers cannot be predictors")
        forbidden = ("lamp1", "outcome", "coloc", "manders", "red_lamp")
        if any(any(token in name.lower() for token in forbidden) for name in names):
            raise ValueError("LAMP1-derived and joint-channel variables cannot enter predictor blocks")
        if len(set(self.construction_animals)) != len(self.construction_animals):
            raise ValueError("construction_animals must be unique")

    def imaging_features(self):
        return [name for block in ("F", "Popt", "PPA") for name in self.feature_blocks[block]]

    def model_inputs(self):
        b = list(self.baseline)
        f, p, a = (self.feature_blocks[key] for key in ("F", "Popt", "PPA"))
        return {"M1": b, "M2": b + f, "M3": b + p, "M4": b + f + p, "M5": b + f + p + a}


def load_config(path):
    values = json.loads(Path(path).read_text(encoding="utf-8"))
    values["network"] = NetworkConfig(**values.get("network", {}))
    for key in ("baseline", "construction_animals", "k_candidates"):
        if key in values:
            values[key] = tuple(values[key])
    config = Config(**values)
    config.validate()
    return config
