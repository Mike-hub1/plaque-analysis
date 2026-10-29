"""Configuration for fluorescence feature measurements."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from plaque_analysis.common.paths import Data, OUTPUT_ROOT


DEFAULT_CZI_PATH = Path(Data) / "sample.czi"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "fluorescence"

DEFAULT_ABETA_CHANNEL = 0   # 用户染色说明: 561 nm = Aβ/CR


DEFAULT_LAMP1_CHANNEL = 2   # 用户染色说明: 405 nm = LAMP1


DEFAULT_TAU_CHANNEL = 1     # 用户染色说明: 488 nm = tau；仅用于原始图展示


VERSION = "7.4.1"


FIGURE_FONT = "Arial"


ID_COLUMNS = ["AnimalID", "SectionID", "FieldID", "PlaqueID"]


TEXTURE_PROPERTIES = ("contrast", "homogeneity", "entropy", "ASM", "energy", "correlation")


@dataclass(frozen=True)
class MeasurementConfig:
    window_radius_um: float = 10.0
    ring_width_um: float | None = None     # None -> 2*max(sx,sy)，保持至少两个像素
    edge_width_um: float | None = None
    angular_sectors: int = 12
    min_object_pixels: int = 9
    min_glcm_pairs: int = 16
    texture_levels: int = 8
    texture_bin_edges: tuple | None = None
    texture_offsets: tuple = ((1, 0), (0, 1), (1, 1), (-1, 1))  # (dx,dy)
    max_saturated_fraction: float = 0.0     # 保守起点：任一上限像素使强度型主特征无效
    random_state: int = 42
    n_clusters: int = 5
    texture_extra_steps: tuple = (2, 3)    # 原生像素位移；每方向另记实际 μm，不能混称同一尺度
    min_gradient_centres: int = 9
    parameters_frozen: bool = False

    def resolved(self, spacing):
        import numpy as np
        sy, sx = spacing
        ring = self.ring_width_um if self.ring_width_um is not None else 2*max(spacing)
        edge = self.edge_width_um if self.edge_width_um is not None else 2*max(spacing)
        if not all(np.isfinite(v) for v in (self.window_radius_um, ring, edge)) or self.window_radius_um <= 0 or ring < max(spacing) or edge <= 0:
            raise ValueError("窗口/边缘带须为正；环宽不得小于最大像素间距。")
        if not 4 <= self.angular_sectors <= 72 or self.min_object_pixels < 2:
            raise ValueError("扇区数需为 4–72；最小对象像素数需 >=2。")
        if not 2 <= self.texture_levels <= 64 or self.min_glcm_pairs < 1:
            raise ValueError("纹理等级需为 2–64；最小有效像素对数需为正。")
        if not np.isfinite(self.max_saturated_fraction) or not 0 <= self.max_saturated_fraction <= 1:
            raise ValueError("最大允许上限像素比例须在 [0,1] 内。")
        if not 1 <= self.n_clusters <= 5 or not 0 <= self.random_state < 2**32:
            raise ValueError("五级强度分层数需为 1–5；随机种子超出范围。")
        if self.min_gradient_centres < 3 or tuple(self.texture_extra_steps) != (2, 3):
            raise ValueError("梯度至少需 3 个有效中心；v7 固定输出原生 2/3 像素位移纹理。")
        if self.texture_bin_edges is not None:
            e = np.asarray(self.texture_bin_edges, float)
            if len(e) < 3 or not np.all(np.isfinite(e)) or np.any(np.diff(e) <= 0):
                raise ValueError("固定分箱边界需至少 3 个有限、严格递增的数。")
        return replace(self, ring_width_um=float(ring), edge_width_um=float(edge))

