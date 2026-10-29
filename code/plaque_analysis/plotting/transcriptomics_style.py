"""Shared manuscript class colours and physical-coordinate display helpers."""

import numpy as np

CELL_COLORS = {
    "Excitatory neuron": "#2F5DA8", "Inhibitory neuron": "#56B4E9",
    "Oligodendrocyte": "#1B9E77", "OPC": "#A6D96A", "Astrocyte": "#E69F00",
    "Microglia": "#D55E00", "Endothelial": "#CC3366", "Mural": "#E4A6B7",
    "VLMC": "#8C564B", "Ependymal": "#7B3294", "Choroid plexus": "#C2A5CF",
    "Low-confidence / ambiguous": "#C8C8C8",
}
ABBREVIATIONS = dict(zip(CELL_COLORS, ("EX", "IN", "OLI", "OPC", "AST", "MIC",
    "END", "MUR", "VLMC", "EPEN", "CHOR", "AMB")))
CLASS_COLORS = {"S1": "#C6972E", "S2": "#3186AD", "S3": "#A55C83"}
REGION_COLORS = {"CTX": "#A9C7E1", "HPF": "#BBA7D0", "STR": "#EAB49F",
    "TH": "#AECFD0", "HY": "#D2CF98", "Transition/undetermined": "#E6E6E6"}
CONFIDENCE_COLORS = {"high": "#9999CC", "medium": "#F4CA87", "low": "#D98E98"}


def display_coordinates(frame, settings):
    """Apply one rigid orientation transform to a/c/e; input units remain µm.

    Set this transform from image orientation so the ventral side faces up.
    Scaling/shearing is rejected to retain physical distances and scale bars.
    """
    matrix = np.asarray(settings["coordinate_transform"], float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[2], [0, 0, 1])
            or not np.allclose(matrix[:2, :2].T @ matrix[:2, :2], np.eye(2), atol=1e-8)):
        raise ValueError("display.coordinate_transform must be a rigid 2-D homogeneous transform")
    xy = frame[["x_um", "y_um"]].to_numpy(float)
    return (np.column_stack((xy, np.ones(len(xy)))) @ matrix.T)[:, :2] / 1000


def spatial_axes(ax, settings):
    ax.set_aspect("equal")
    if settings.get("invert_y", True):
        ax.invert_yaxis()
    ax.set(xlabel="x (mm)", ylabel="y (mm)")
    ax.spines[["top", "right"]].set_visible(False)


def add_scale_bar(ax, length_mm=1):
    from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
    ax.add_artist(AnchoredSizeBar(ax.transData, length_mm, f"{length_mm:g} mm",
        loc="lower left", pad=.4, color="#253F48", frameon=False, size_vertical=.005))
