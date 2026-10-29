"""Command-line interface for fluorescence measurements."""
from __future__ import annotations

from plaque_analysis.settings.fluorescence import DEFAULT_ABETA_CHANNEL, DEFAULT_CZI_PATH, DEFAULT_LAMP1_CHANNEL, DEFAULT_OUTPUT_DIR, DEFAULT_TAU_CHANNEL
from pathlib import Path
import argparse


def comma_floats(text):
    try:
        return tuple(float(x.strip()) for x in text.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("请输入逗号分隔的数字。") from exc


def build_argument_parser():
    p = argparse.ArgumentParser(
        description="Fluorescence feature extraction, spatial measurements and plotting (CPU)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--czi-path", type=Path, default=DEFAULT_CZI_PATH,
                   help="输入 CZI 文件；修改 common/paths.py 中的 Data，或用此参数指定路径")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help="输出目录；默认位于 Github/outputs/fluorescence，自定义相对路径以当前工作目录为准")
    p.add_argument("--abeta-channel", type=int, default=DEFAULT_ABETA_CHANNEL)
    p.add_argument("--lamp1-channel", type=int, default=DEFAULT_LAMP1_CHANNEL)
    p.add_argument("--tau-channel", type=int, default=DEFAULT_TAU_CHANNEL,
                   help="488 nm 的 tau 原始图通道；不加入现有 Aβ/LAMP1 特征分析")
    p.add_argument("--scene-index", type=int, default=0)
    p.add_argument("--time-index", type=int, default=0)
    p.add_argument("--z-index", type=int, default=0)
    p.add_argument("--pixel-size-x-um", type=float)
    p.add_argument("--pixel-size-y-um", type=float)
    p.add_argument("--abeta-effective-bits", type=int)
    p.add_argument("--lamp1-effective-bits", type=int)
    p.add_argument("--tau-effective-bits", type=int,
                   help="tau 的真实有效位深，例如 uint16 容器中的 12-bit；仅影响原始图显示与上限记录")
    p.add_argument("--background-method", choices=["percentile", "none", "rolling-ball"], default="percentile")
    p.add_argument("--background-percentile", type=float, default=5)
    p.add_argument("--rolling-ball-radius", type=float, default=50, help="分割预处理候选背景半径，pixel")
    p.add_argument("--foreground-method", choices=["multi-otsu-hysteresis", "otsu", "positive"],
                   default="multi-otsu-hysteresis")
    p.add_argument("--foreground-classes", type=int, choices=range(3, 6), default=4)
    p.add_argument("--abeta-min-signal", type=float)
    p.add_argument("--lamp1-min-signal", type=float)
    for name, help_text in (
        ("background-roi", "独立核实的背景组织 ROI，非零像素作为背景"),
        ("valid-mask", "有效观测组织掩膜，非零为有效"),
        ("abeta-mask", "独立于 LAMP1 定义的 Aβ 整数对象标签；二值文件按连通域处理"),
        ("lamp1-mask", "独立 LAMP1 分割掩膜"),
        ("reference-mask-abeta", "用于检验 Aβ 分割准确度的独立人工掩膜"),
        ("reference-mask-lamp1", "用于检验 LAMP1 分割准确度的独立人工掩膜"),
        ("reference-abeta", "未作显示拉伸的原始 Aβ 参考 TIFF，用于逐像素验证 CZI"),
        ("reference-lamp1", "未作显示拉伸的原始 LAMP1 参考 TIFF"),
        ("repeat-table", "真实重复成像长表: unit_id,repeat_id,feature,value"),
        ("split-manifest", "下游分折清单: AnimalID,SectionID,FieldID,PlaqueID,split")):
        p.add_argument("--"+name, type=Path, help=help_text)
    for name in ("animal-id", "section-id", "field-id", "plaque-id"):
        p.add_argument("--"+name, help="可追溯标识；未知时保留缺失，不从文件名假定动物独立性")
    p.add_argument("--age-months", type=float)
    p.add_argument("--region")
    p.add_argument("--staining-batch")
    p.add_argument("--acquisition-batch")
    p.add_argument("--window-radius-um", type=float, default=10,
                   help="Aβ 边界外固定窗口的探索性起点；需参考集验证后冻结")
    p.add_argument("--ring-width-um", type=float, help="默认两像素；不允许亚像素环带")
    p.add_argument("--edge-width-um", type=float, help="Aβ 几何边缘带宽；默认两像素")
    p.add_argument("--angular-sectors", type=int, default=12)
    p.add_argument("--min-object-pixels", type=int, default=9)
    p.add_argument("--min-glcm-pairs", type=int, default=16)
    p.add_argument("--min-gradient-centres", type=int, default=9,
                   help="内部梯度至少需要的有效中心数；中心及上下左右邻居均须在 Aβ 掩膜内")
    p.add_argument("--texture-levels", type=int, default=8)
    p.add_argument("--texture-bin-edges", type=comma_floats,
                   help="预先固定绝对分箱；缺省采用对象内相对 min/max，非跨样本浓度等级")
    p.add_argument("--max-saturated-fraction", type=float, default=0,
                   help="强度型特征允许的上限像素比例；默认任一上限像素标记失效")
    p.add_argument("--n-clusters", type=int, choices=range(1, 6), default=5, help="辅助强度分层，不作为生物学亚型")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--dpi", type=int, default=600, help="绘图 PNG 的输出 DPI；不改变显微图原生采样")
    p.add_argument("--parameters-frozen", action="store_true", help="仅在独立验证后标记参数冻结；不等于脚本验证生物学有效性")
    p.add_argument("--skip-sensitivity", action="store_true")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--redraw-figures", action="store_true",
                   help="只读 --out-dir 中既有矩阵/CSV 重绘，不重新分割、聚类或改变分析值；需 --overwrite")
    p.add_argument("--redraw-manuscript-only", action="store_true",
                   help="仅重绘正文整合图、九个子图和 A4 校样；与 --redraw-figures --overwrite 同用")
    p.add_argument("--skip-raw-channel-figures", action="store_true",
                   help="仅关闭新增原始通道导出，不影响原有九类分析图；无需读取 tau")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--check-inputs", action="store_true")
    return p


def main(argv=None):
    parser = build_argument_parser()
    args = parser.parse_args(argv)
    if args.dpi <= 0:
        parser.error("--dpi 必须为正。")
    if args.redraw_manuscript_only and not args.redraw_figures:
        parser.error("--redraw-manuscript-only 必须与 --redraw-figures 同用。")
    if (args.pixel_size_x_um is None) != (args.pixel_size_y_um is None):
        parser.error("手动标定时必须同时指定 x/y 像素间距。")
    try:
        if args.redraw_figures:
            from plaque_analysis.workflows.fluorescence_redraw import redraw_existing_figures
            redraw_existing_figures(args)
        else:
            if args.czi_path is None or args.czi_path == Path("your project/dataset/sample.czi"):
                parser.error("输入路径尚未配置：请修改 common/paths.py 中的 Data，或使用 --czi-path PATH。")
            if not args.czi_path.expanduser().is_file():
                parser.error(f"输入 CZI 文件不存在：{args.czi_path}")
            from plaque_analysis.workflows.fluorescence import process_two_images
            process_two_images(args)
    except (ValueError, OSError, RuntimeError, ImportError) as exc:
        parser.error(str(exc))
    return 0

