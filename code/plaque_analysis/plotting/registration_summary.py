"""Class-level mapping availability and the same LOO errors used in panel c."""

import numpy as np


def draw_d(source):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 6), sharey=True, layout="constrained")
    positions = np.arange(len(source))
    axes[0].barh(positions, source.total_n, color="#76BCA6")
    axes[0].set_yticks(positions, source.broad_celltype)
    axes[0].invert_yaxis()
    axes[0].set(xlabel="Nuclei (n)", title="d  All nuclei by cell class")
    lower = source.invalid_percent-source.wilson95_low_percent
    upper = source.wilson95_high_percent-source.invalid_percent
    axes[1].errorbar(source.invalid_percent, positions, xerr=np.maximum(0, np.vstack((lower, upper))),
                    fmt="o", ms=3, color="#2AA789", lw=.8)
    axes[1].set(xlabel="Unavailable mappings (%)", title="Wilson 95% interval")
    for position, row in zip(positions, source.itertuples()):
        values = np.array([row.registration_error_um_p05, row.registration_error_um_p25,
            row.registration_error_um_median, row.registration_error_um_p75, row.registration_error_um_p95])
        if np.isfinite(values).all():
            axes[2].plot(values[[0, 4]], [position, position], color="#2AA789", lw=.8)
            axes[2].plot(values[[1, 3]], [position, position], color="#2AA789", lw=5, alpha=.7)
            axes[2].plot(values[2], position, "|", color="#263D40", ms=9)
            axes[2].annotate(f"{values[2]:.2f}", (values[2], position),
                             xytext=(0, 7), textcoords="offset points",
                             ha="center", va="bottom", fontsize=7)
        else:
            axes[2].text(.02, position, "No available mapping", transform=axes[2].get_yaxis_transform(), fontsize=7)
    axes[2].set(xlabel="Interpolated LOO error (µm)", title="Median · IQR · 5–95%")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)
    return fig
