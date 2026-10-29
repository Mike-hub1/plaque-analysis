"""Explicit parameters for the legacy colour-image measurement workflow."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class AreaConfig:
    """Pixel sizes are in micrometres; other lengths are in pixels.

    Defaults preserve the original measurement parameters. Pixel sizes must be
    checked against acquisition metadata before measuring another image.
    """

    small_region_threshold: int = 25
    merge_distance_threshold: float = 10
    top_n: int = 10
    pixel_size_x: float = 1.66
    pixel_size_y: float = 1.66
    min_white_region_size: int = 0
    search_range_multiplier: float = 1.0
    dilation_size: int = 15
    inner_erosion: int = 3
    interactive: bool = False
    seed: int = 0

    def __post_init__(self):
        for name in ("pixel_size_x", "pixel_size_y", "search_range_multiplier"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("small_region_threshold", "top_n"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("min_white_region_size", "dilation_size", "inner_erosion"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if not math.isfinite(self.merge_distance_threshold) or self.merge_distance_threshold < 0:
            raise ValueError("merge_distance_threshold must be finite and nonnegative")
        if (not isinstance(self.seed, int) or isinstance(self.seed, bool)
                or not 0 <= self.seed <= 2**31 - 1):
            raise ValueError("seed must be an integer between 0 and 2**31 - 1")
