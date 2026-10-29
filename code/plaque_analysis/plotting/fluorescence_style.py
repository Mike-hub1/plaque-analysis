"""Figure 3: plot style."""
from __future__ import annotations

from plaque_analysis.settings.fluorescence import FIGURE_FONT
from plaque_analysis.plotting.fluorescence_layout import save_individual_panels


def configure_plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager.findfont(FIGURE_FONT, fallback_to_default=False)
    plt.rcParams.update({
        "font.family": FIGURE_FONT, "font.size": 6.5, "axes.titlesize": 7,
        "axes.labelsize": 6.3, "xtick.labelsize": 5.8, "ytick.labelsize": 5.8,
        "legend.fontsize": 5.6, "svg.fonttype": "none", "pdf.fonttype": 42,
        "mathtext.fontset": "custom", "mathtext.rm": FIGURE_FONT,
        "mathtext.it": FIGURE_FONT+":italic", "mathtext.bf": FIGURE_FONT+":bold",
        "mathtext.sf": "Symbol", "axes.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "xtick.major.width": .5, "ytick.major.width": .5,
        "text.color": "#242429", "axes.labelcolor": "#242429",
        "xtick.color": "#242429", "ytick.color": "#242429",
        "legend.frameon": False,
        "figure.facecolor": "white", "savefig.facecolor": "white",
    })
    return plt


NATURE_SYMBOL_CHARACTERS = frozenset("βμΔθ→×±°")


def nature_symbol_text(value):
    if "$" in value:
        return value
    return "".join(r"$\mathsf{"+c+r"}$" if c in NATURE_SYMBOL_CHARACTERS else c for c in value)


def apply_nature_text_policy(fig):
    """仅正文组版：英文 Arial；希腊字母/专用符号采用可编辑 Symbol 字形。"""
    from matplotlib.text import Text
    from matplotlib.ticker import FuncFormatter
    from matplotlib import font_manager
    font_manager.findfont("Symbol", fallback_to_default=False)
    fig.canvas.draw()  # 先生成实际刻度；不遗漏 colorbar 或图例里的文字。
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            labels = axis.get_majorticklabels()
            if any(any(c in NATURE_SYMBOL_CHARACTERS for c in label.get_text()) for label in labels):
                # 仅修改临时 tick Text 会在下一次 draw 时被 Formatter 还原。
                formatter = axis.get_major_formatter()
                axis.set_major_formatter(FuncFormatter(
                    lambda value, pos, original=formatter: nature_symbol_text(original(value, pos))))
    for artist in fig.findobj(match=Text):
        original = artist.get_text()
        if "$" in original:
            continue  # 已处理的文字不能重复套入数学文本。
        if any(c in NATURE_SYMBOL_CHARACTERS for c in original):
            artist.set_text(nature_symbol_text(original))
            artist.set_math_fontfamily("custom")


def save_figure(fig, base, dpi):
    # Exact physical size, no bbox='tight' rescaling. Pixel arrays remain unmodified.
    base.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".svg", ".png"):
        fig.savefig(str(base)+suffix, dpi=dpi)
    if getattr(fig, "_panel_order", None):
        save_individual_panels(fig, base, dpi)
    import matplotlib.pyplot as plt
    plt.close(fig)


FIGURE_COLORS = {"A": "#3C759C", "L": "#C77255", "both": "#88719A",
                 "ink": "#303039", "neutral": "#CBCDD1", "light": "#F1F2F3",
                 "ceiling": "#B8994F", "withheld": "#C7A565"}


RAW_CHANNEL_FIGURE_FOLDER = "00_CZI原始图"


RAW_CHANNEL_NAMES = ("Aβ", "LAMP1", "tau")


RAW_CHANNEL_RGB = ((0., 0., 1.), (1., 0., 0.), (0., 1., 0.))


RAW_CHANNEL_VIEWS = (("Aβ", (0,)), ("LAMP1", (1,)), ("tau", (2,)),
                     ("Aβ_LAMP1_双通道合并", (0, 1)),
                     ("Aβ_LAMP1_tau_三通道合并", (0, 1, 2)))

