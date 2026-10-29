"""Analysis panels from newly computed source tables; final artwork is kept separately."""
from pathlib import Path
import numpy as np
import pandas as pd


def _save(fig, folder, name):
    import matplotlib.pyplot as plt
    fig.savefig(Path(folder) / f"{name}.svg", bbox_inches="tight")
    fig.savefig(Path(folder) / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_prediction_tables(root, destination, config):
    import matplotlib.pyplot as plt
    from plaque_analysis.models.metrics import animal_bootstrap_line
    prediction = Path(root) / "prediction"
    if not (prediction / "predictions.csv").is_file(): return
    table = pd.read_csv(prediction / "predictions.csv", dtype={"animal_id": str})
    fig, axes = plt.subplots(1, 2, figsize=(7, 3), sharex=True, sharey=True)
    for axis, model in zip(axes, ("M1", "M5")):
        part = table[table.model == model]
        band = animal_bootstrap_line(part, config.bootstrap_repeats, config.seed)
        band.to_csv(prediction / f"{model}_display_line.csv", index=False)
        axis.scatter(part.observed, part.predicted, s=5, alpha=.35)
        axis.plot(band.observed, band.fitted, color="black")
        axis.fill_between(band.observed, band.ci_low, band.ci_high, alpha=.15)
        axis.set(xlabel="Observed log1p LAMP1 area", ylabel="Predicted log1p LAMP1 area", title=model)
    _save(fig, destination, "panel_c_predictions")
    errors = pd.read_csv(prediction / "animal_mae.csv").pivot(index="animal_id", columns="model", values="mae")
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(np.arange(len(errors.columns)), errors.to_numpy().T, color=".75", lw=.6)
    ax.scatter(np.arange(len(errors.columns)), errors.mean(), color="black", marker="D")
    ax.set(xticks=np.arange(len(errors.columns)), xticklabels=errors.columns, ylabel="Animal MAE")
    _save(fig, destination, "panel_d_animal_mae")
    gains = pd.read_csv(prediction / "paired_gain_summary.csv")
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.barh(gains.comparator, gains.relative_reduction_percent)
    ax.set(xlabel="MAE reduction relative to comparator (%)")
    _save(fig, destination, "panel_i_relative_gain")


def plot_atlas_tables(root, destination):
    import matplotlib.pyplot as plt
    atlas = Path(root) / "atlas"
    if not (atlas / "display_pca.csv").is_file(): return
    data = pd.read_csv(atlas / "display_pca.csv")
    fig, ax = plt.subplots(figsize=(4, 3.5))
    for label, part in data[data.subset == "construction"].groupby("class"):
        ax.scatter(part.PC1, part.PC2, s=7, alpha=.6, label=label)
    ax.set(xlabel="PC1 (display only)", ylabel="PC2 (display only)")
    ax.legend(frameon=False)
    _save(fig, destination, "panel_e_reference_projection")
    if (atlas / "animal_class_residuals.csv").is_file():
        residual = pd.read_csv(atlas / "animal_class_residuals.csv").pivot(index="animal_id", columns="class", values="mean_residual")
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.plot(np.arange(len(residual.columns)), residual.to_numpy().T, marker="o", lw=.6)
        ax.axhline(0, color=".7", lw=.6)
        ax.set(xticks=np.arange(len(residual.columns)), xticklabels=residual.columns, ylabel="Animal mean LAMP1 residual")
        _save(fig, destination, "panel_g_validation_residuals")


def plot_stability_tables(root, destination):
    import matplotlib.pyplot as plt
    folder = Path(root) / "stability"
    if not (folder / "heldout_stability.csv").is_file(): return
    data = pd.read_csv(folder / "heldout_stability.csv")
    ordered = ["F", "Popt", "PPA", "F+Popt+PPA"]
    fig, ax = plt.subplots(figsize=(5, 3))
    paired = data.pivot(index="repeat", columns="feature_set", values="heldout_ari")[ordered]
    ax.plot(range(4), paired.to_numpy().T, color=".8", lw=.4)
    ax.boxplot([paired[c].to_numpy() for c in ordered], positions=range(4), showfliers=False)
    ax.set(xticks=range(4), xticklabels=ordered, ylabel="Held-out ARI")
    _save(fig, destination, "panel_h_reference_stability")
    posterior = pd.read_csv(folder / "mean_posterior.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    artist = ax.imshow(posterior.to_numpy(), vmin=0, vmax=1, cmap="Blues")
    ax.set(xticks=range(len(posterior.columns)), xticklabels=posterior.columns,
           yticks=range(len(posterior)), yticklabels=posterior.index,
           xlabel="Reference class", ylabel="Inferred class")
    fig.colorbar(artist, ax=ax, label="Mean posterior probability")
    _save(fig, destination, "panel_f_mean_posteriors")


def plot_results(root, config):
    import matplotlib
    matplotlib.use("Agg")
    destination = Path(root) / "panels"
    destination.mkdir(parents=True, exist_ok=True)
    plot_prediction_tables(root, destination, config)
    plot_atlas_tables(root, destination)
    plot_stability_tables(root, destination)
