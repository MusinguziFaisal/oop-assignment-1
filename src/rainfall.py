"""Mini-Project 4: Rainfall Pattern & Crop Suitability Analyser."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.signal import find_peaks

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Illustrative monthly rainfall (mm) from the brief.
RAINFALL_DATA: Dict[str, List[float]] = {
    "Kampala": [120, 140, 180, 200, 220, 180, 90, 70, 60, 100, 110, 130],
    "Gulu": [8, 25, 75, 160, 190, 145, 170, 215, 175, 150, 60, 15],
    "Mbarara": [70, 85, 120, 140, 90, 25, 20, 55, 100, 125, 120, 90],
}


class Region:
    """Monthly rainfall (mm, Jan-Dec) for one region."""

    def __init__(self, name: str, rainfall_mm: Sequence[float]) -> None:
        r = np.asarray(rainfall_mm, dtype=float)
        if r.shape != (12,):
            raise ValueError("rainfall must have exactly 12 monthly values")
        if not np.all(np.isfinite(r)) or np.any(r < 0):
            raise ValueError("rainfall must be finite and non-negative")
        self.name = name
        self.rainfall = r

    def __repr__(self) -> str:
        return f"Region({self.name!r}, annual={self.annual_total():.0f} mm)"

    def annual_total(self) -> float:
        """Total annual rainfall (mm)."""
        return float(self.rainfall.sum())

    def mean(self) -> float:
        """Mean monthly rainfall (mm)."""
        return float(self.rainfall.mean())

    def wettest_month(self) -> str:
        return MONTHS[int(np.argmax(self.rainfall))]

    def driest_month(self) -> str:
        return MONTHS[int(np.argmin(self.rainfall))]

    def cv(self) -> float:
        """Coefficient of variation of monthly rainfall (population std / mean)."""
        m = self.rainfall.mean()
        if m == 0:
            raise ValueError("mean rainfall is zero: CV undefined")
        return float(self.rainfall.std() / m)


@dataclass(frozen=True)
class CropRule:
    """Suitable monthly rainfall range ``[low_mm, high_mm]`` for a crop.

    ``source`` documents where the thresholds come from.
    """

    crop: str
    low_mm: float
    high_mm: float
    source: str = ""

    def __post_init__(self) -> None:
        if not 0 <= self.low_mm < self.high_mm:
            raise ValueError("need 0 <= low_mm < high_mm")

    def classify(self, rain_mm: float) -> str:
        """``Drought risk`` below the range, ``Waterlogging risk`` above, else ``Good``."""
        if rain_mm < 0:
            raise ValueError("rainfall cannot be negative")
        if rain_mm < self.low_mm:
            return "Drought risk"
        if rain_mm > self.high_mm:
            return "Waterlogging risk"
        return "Good"

    def classify_region(self, region: Region) -> List[str]:
        """Classification for all 12 months."""
        return [self.classify(x) for x in region.rainfall]


def default_crop_rules() -> List[CropRule]:
    """Maize, beans, coffee. Thresholds derived from FAO and UCDA figures.

    * Maize: FAO says 500-800 mm per growing period; assuming a 4-month season
      this is 125-200 mm per month.
    * Beans: FAO 300-500 mm per growing period; assuming a 3-month season
      this is 100-167 mm per month.
    * Coffee (robusta): UCDA 1,200-1,800 mm per year, well distributed over about
      9 months; lower bound 1,200/12 = 100 mm, upper bound 1,800/9 = 200 mm.

    Season lengths and the use of the upper bound as a waterlogging threshold
    are the analyst's assumptions, not figures from the sources.
    """
    fao = "FAO, 'Water and soil requirements', Table 2 (https://www.fao.org/3/u3160e/u3160e04.htm)"
    ucda = "UCDA coffee handbook, s.2.4 (https://www.ugandacoffee.go.ug/file-download/download/public/42)"
    return [
        CropRule("maize", 500 / 4, 800 / 4, fao + "; 4-month season assumed"),
        CropRule("beans", 300 / 3, 500 / 3, fao + "; 3-month season assumed"),
        CropRule("coffee", 1200 / 12, 1800 / 9, ucda + "; lower=annual/12, upper=annual/9"),
    ]


def month_labels(region: Region, rules: Sequence[CropRule]) -> List[str]:
    """One label per month: ``Good for a, b`` or the dominant risk if no crop is good."""
    labels: List[str] = []
    for m in range(12):
        statuses = {r.crop: r.classify(region.rainfall[m]) for r in rules}
        good = [c for c, s in statuses.items() if s == "Good"]
        if good:
            labels.append("Good for " + ", ".join(good))
        elif all(s == "Drought risk" for s in statuses.values()):
            labels.append("Drought risk")
        elif all(s == "Waterlogging risk" for s in statuses.values()):
            labels.append("Waterlogging risk")
        else:
            labels.append("Mixed risk")
    return labels


# ---------------------------------------------------------------------------
# Similarity measures
# ---------------------------------------------------------------------------
def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """cos(theta) = a.b / (|a||b|), computed with NumPy (not ``math.cos``)."""
    x, y = np.asarray(a, float), np.asarray(b, float)
    if x.shape != y.shape:
        raise ValueError("vectors must have the same length")
    nx, ny = np.linalg.norm(x), np.linalg.norm(y)
    if nx == 0 or ny == 0:
        raise ValueError("cosine similarity undefined for a zero vector")
    return float(np.dot(x, y) / (nx * ny))


def pearson(a: Sequence[float], b: Sequence[float]) -> float:
    """Pearson correlation coefficient."""
    return float(np.corrcoef(np.asarray(a, float), np.asarray(b, float))[0, 1])


def euclidean(a: Sequence[float], b: Sequence[float]) -> float:
    """Euclidean distance."""
    return float(np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


def pairwise_matrix(regions: Sequence[Region], func) -> np.ndarray:
    """Symmetric matrix of ``func(region_i.rainfall, region_j.rainfall)``."""
    n = len(regions)
    out = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            out[i, j] = func(regions[i].rainfall, regions[j].rainfall)
    return out


# ---------------------------------------------------------------------------
# Rainy-season detection
# ---------------------------------------------------------------------------
def detect_rainy_seasons(region: Region, min_rel_prominence: float = 0.25) -> List[str]:
    """Peak months found with ``scipy.signal.find_peaks``.

    The year is treated as circular (the series is tiled three times and the
    middle copy kept) so a season straddling Dec/Jan is not cut. A peak counts
    only if its prominence is at least ``min_rel_prominence`` x the wettest month.
    """
    if not 0 <= min_rel_prominence <= 1:
        raise ValueError("min_rel_prominence must be in [0, 1]")
    r = region.rainfall
    tiled = np.tile(r, 3)
    peaks, _ = find_peaks(tiled, prominence=min_rel_prominence * r.max())
    mid = sorted({int(p) - 12 for p in peaks if 12 <= p < 24})
    return [MONTHS[m] for m in mid]


def rainfall_regime(region: Region, min_rel_prominence: float = 0.25) -> str:
    """``unimodal`` (one rainy season), ``bimodal`` (two) or ``multimodal``/``no clear season``."""
    k = len(detect_rainy_seasons(region, min_rel_prominence))
    return {0: "no clear season", 1: "unimodal", 2: "bimodal"}.get(k, "multimodal")
