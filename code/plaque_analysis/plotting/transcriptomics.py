"""Draw all six panels from one set of computed manuscript-method results."""

from pathlib import Path
from plaque_analysis.plotting.registered_space import draw_a, draw_c, draw_e
from plaque_analysis.plotting.umap import draw_b
from plaque_analysis.plotting.registration_summary import draw_d
from plaque_analysis.plotting.expression import draw_f
from plaque_analysis.plotting.transcriptomics_style import CLASS_COLORS


def replot_panels(prepared, output_dir, dpi=300):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not isinstance(dpi, int) or dpi < 1:
        raise ValueError("dpi must be a positive integer")
    output = Path(output_dir).resolve() / "regenerated"
    output.mkdir(parents=True, exist_ok=True)
    spatial, settings = prepared["spatial"], prepared["config"]["display"]
    renderers = {"a": lambda: draw_a(spatial, settings), "b": lambda: draw_b(spatial),
                 "c": lambda: draw_c(spatial, settings), "d": lambda: draw_d(prepared["qc"]),
                 "e": lambda: draw_e(spatial, settings),
                 "f": lambda: draw_f(prepared["expression"], CLASS_COLORS)}
    with plt.rc_context({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                        "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8}):
        for letter, render in renderers.items():
            figure = render()
            for extension in ("svg", "png", "pdf"):
                figure.savefig(output / f"Figure6{letter}.{extension}", dpi=dpi, facecolor="white")
            plt.close(figure)
    return {"panels": list(renderers), "formats": ["svg", "png", "pdf"], "dpi": dpi,
            "registration_unit": "µm; identical error estimator in panels c and d",
            "class_source": "fixed imaging model", "final_artwork_modified": False}
