"""Mini-Project 1: UBOS District Population Forecaster.

Contains the ``DistrictPopulation`` data class, an abstract ``Forecaster`` base
class with three concrete models (linear, CAGR/exponential and Fibonacci-ratio),
error metrics, a residual-bootstrap for prediction intervals and the
classroom-planning calculation.
"""
from __future__ import annotations

import copy
import math
import statistics
from abc import ABC, abstractmethod
from typing import Dict, Sequence, Tuple

import numpy as np

PHI = (1 + 5 ** 0.5) / 2  # golden ratio, limit of F(n+1)/F(n)

# ---------------------------------------------------------------------------
# Data (thousands of people, 2015-2024). Kampala/Wakiso/Gulu are from the brief;
# Mbarara and Jinja are ILLUSTRATIVE additions chosen by the student.
# ---------------------------------------------------------------------------
YEARS = list(range(2015, 2025))
DISTRICT_DATA: Dict[str, list] = {
    "Kampala": [1200, 1250, 1300, 1350, 1420, 1500, 1580, 1650, 1720, 1800],
    "Wakiso": [950, 1000, 1070, 1150, 1220, 1300, 1390, 1480, 1570, 1670],
    "Gulu": [320, 330, 345, 360, 375, 390, 410, 430, 455, 480],
    "Mbarara": [480, 495, 512, 530, 548, 568, 588, 610, 632, 655],
    "Jinja": [410, 425, 438, 449, 459, 468, 476, 483, 489, 494],
}


class DistrictPopulation:
    """Annual population estimates (thousands) for one district.

    Parameters
    ----------
    name:
        District name.
    years:
        Strictly increasing calendar years.
    populations:
        Population in thousands, one value per year.

    Raises
    ------
    ValueError
        If lengths differ, the series is empty, a value is negative/non-finite,
        or years are not strictly increasing.
    """

    def __init__(self, name: str, years: Sequence[int], populations: Sequence[float]) -> None:
        years_arr = np.asarray(years, dtype=int)
        pops_arr = np.asarray(populations, dtype=float)
        if years_arr.ndim != 1 or pops_arr.ndim != 1:
            raise ValueError("years and populations must be one-dimensional")
        if len(years_arr) != len(pops_arr):
            raise ValueError(
                f"years ({len(years_arr)}) and populations ({len(pops_arr)}) must have equal length"
            )
        if len(years_arr) == 0:
            raise ValueError("series must not be empty")
        if not np.all(np.isfinite(pops_arr)):
            raise ValueError("populations must be finite numbers")
        if np.any(pops_arr < 0):
            raise ValueError("populations must not be negative")
        if np.any(np.diff(years_arr) <= 0):
            raise ValueError("years must be strictly increasing")
        self.name = name
        self.years = years_arr
        self.populations = pops_arr

    # -- dunder methods -----------------------------------------------------
    def __len__(self) -> int:
        return len(self.years)

    def __repr__(self) -> str:
        return (
            f"DistrictPopulation(name={self.name!r}, years={int(self.years[0])}-"
            f"{int(self.years[-1])}, n={len(self)}, last={self.populations[-1]:.0f}k)"
        )

    # -- descriptive statistics --------------------------------------------
    def stats_statistics_module(self) -> Dict[str, float]:
        """Mean, median, sample variance/std using the ``statistics`` module."""
        data = self.populations.tolist()
        return {
            "mean": statistics.mean(data),
            "median": statistics.median(data),
            "variance": statistics.variance(data),  # sample variance, n-1
            "std": statistics.stdev(data),
        }

    def stats_numpy(self, ddof: int = 0) -> Dict[str, float]:
        """Mean, median, variance/std using NumPy. ``ddof=0`` is NumPy's default."""
        return {
            "mean": float(np.mean(self.populations)),
            "median": float(np.median(self.populations)),
            "variance": float(np.var(self.populations, ddof=ddof)),
            "std": float(np.std(self.populations, ddof=ddof)),
        }

    # -- growth ------------------------------------------------------------
    def yoy_growth(self) -> np.ndarray:
        """Year-on-year growth rates as fractions (length n-1)."""
        prev = self.populations[:-1]
        if np.any(prev == 0):
            raise ValueError("cannot compute growth from a zero population")
        return np.diff(self.populations) / prev

    def cagr(self) -> float:
        """Compound annual growth rate over the whole series."""
        periods = len(self) - 1
        if periods < 1:
            raise ValueError("need at least two observations for CAGR")
        first, last = self.populations[0], self.populations[-1]
        if first <= 0:
            raise ValueError("first population must be positive for CAGR")
        return float((last / first) ** (1 / periods) - 1)

    # -- train / test split ------------------------------------------------
    def split(self, last_train_year: int) -> Tuple["DistrictPopulation", "DistrictPopulation"]:
        """Split into (train, test) at ``last_train_year`` (inclusive)."""
        mask = self.years <= last_train_year
        if mask.all() or not mask.any():
            raise ValueError("split year leaves an empty train or test set")
        return (
            DistrictPopulation(self.name, self.years[mask], self.populations[mask]),
            DistrictPopulation(self.name, self.years[~mask], self.populations[~mask]),
        )


# ---------------------------------------------------------------------------
# Error metrics
# ---------------------------------------------------------------------------
def mae(actual: Sequence[float], pred: Sequence[float]) -> float:
    """Mean absolute error."""
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.mean(np.abs(a - p)))


def rmse(actual: Sequence[float], pred: Sequence[float]) -> float:
    """Root mean squared error."""
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mape(actual: Sequence[float], pred: Sequence[float]) -> float:
    """Mean absolute percentage error (in %). Actual values must be non-zero."""
    a, p = np.asarray(actual, float), np.asarray(pred, float)
    if np.any(a == 0):
        raise ValueError("MAPE undefined when an actual value is zero")
    return float(np.mean(np.abs((a - p) / a)) * 100)


# ---------------------------------------------------------------------------
# Forecasters
# ---------------------------------------------------------------------------
class Forecaster(ABC):
    """Abstract base class: ``fit`` on a series, then ``predict(horizon)``."""

    name: str = "Forecaster"

    def __init__(self) -> None:
        self._fitted = False
        self.years_: np.ndarray = np.array([])
        self.values_: np.ndarray = np.array([])

    def fit(self, years: Sequence[int], values: Sequence[float]) -> "Forecaster":
        """Fit the model to ``values`` observed at consecutive ``years``."""
        y = np.asarray(values, float)
        x = np.asarray(years, float)
        if len(x) != len(y) or len(y) < 3:
            raise ValueError("need equal-length years/values with at least 3 points")
        self.years_, self.values_ = x, y
        self._fit(x, y)
        self._fitted = True
        return self

    def predict(self, horizon: int) -> np.ndarray:
        """Forecast ``horizon`` steps beyond the last training year."""
        if not self._fitted:
            raise RuntimeError("call fit() before predict()")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        return self._predict(int(horizon))

    def fitted_values(self) -> np.ndarray:
        """In-sample fitted values aligned with the training years."""
        if not self._fitted:
            raise RuntimeError("call fit() before fitted_values()")
        return self._fitted_values()

    def residuals(self) -> np.ndarray:
        """Training residuals (actual - fitted)."""
        return self.values_ - self.fitted_values()

    @abstractmethod
    def _fit(self, x: np.ndarray, y: np.ndarray) -> None: ...

    @abstractmethod
    def _predict(self, horizon: int) -> np.ndarray: ...

    @abstractmethod
    def _fitted_values(self) -> np.ndarray: ...


class LinearTrend(Forecaster):
    """Least-squares straight line via ``np.polyfit`` (degree 1)."""

    name = "Linear trend"

    def _fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.slope_, self.intercept_ = np.polyfit(x, y, 1)

    def _predict(self, horizon: int) -> np.ndarray:
        future = self.years_[-1] + np.arange(1, horizon + 1)
        return self.slope_ * future + self.intercept_

    def _fitted_values(self) -> np.ndarray:
        return self.slope_ * self.years_ + self.intercept_


class ExponentialGrowth(Forecaster):
    """Constant-rate growth: ``last * (1 + CAGR)^h`` with CAGR from the training data."""

    name = "Exponential (CAGR)"

    def _fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if y[0] <= 0:
            raise ValueError("first value must be positive")
        self.cagr_ = (y[-1] / y[0]) ** (1 / (len(y) - 1)) - 1

    def _predict(self, horizon: int) -> np.ndarray:
        return self.values_[-1] * (1 + self.cagr_) ** np.arange(1, horizon + 1)

    def _fitted_values(self) -> np.ndarray:
        return self.values_[0] * (1 + self.cagr_) ** np.arange(len(self.values_))


def fibonacci(n: int) -> list:
    """First ``n`` Fibonacci numbers starting 1, 1, 2, 3, ..."""
    if n < 0:
        raise ValueError("n must be non-negative")
    seq: list = []
    a, b = 1, 1
    for _ in range(n):
        seq.append(a)
        a, b = b, a + b
    return seq


class FibonacciRatio(Forecaster):
    """Scale the last value by successive Fibonacci ratios F(k+1)/F(k).

    The raw ratios (1.5, 1.67, 1.6, ...) imply ~60 % growth per year, which is
    absurd for a population. With ``calibrate=True`` (default) the ratio's
    *shape* is kept but its size is rescaled so that the long-run step growth
    equals the mean historical growth ``g``::

        step_h = 1 + g * (r_h - 1) / (PHI - 1),   r_h = F(h+s+1) / F(h+s)

    and ``forecast_h = forecast_{h-1} * step_h``. With ``calibrate=False`` the
    raw ratios multiply the last value directly (the original-cohort method).
    """

    name = "Fibonacci ratio"

    def __init__(self, calibrate: bool = True, offset: int = 3) -> None:
        super().__init__()
        self.calibrate = calibrate
        self.offset = offset

    def _ratios(self, horizon: int) -> np.ndarray:
        fibs = fibonacci(horizon + self.offset + 2)
        return np.array(
            [fibs[k + self.offset] / fibs[k + self.offset - 1] for k in range(1, horizon + 1)]
        )

    def _fit(self, x: np.ndarray, y: np.ndarray) -> None:
        if np.any(y[:-1] == 0):
            raise ValueError("zero value in training data")
        self.g_ = float(np.mean(np.diff(y) / y[:-1]))

    def _predict(self, horizon: int) -> np.ndarray:
        r = self._ratios(horizon)
        steps = 1 + self.g_ * (r - 1) / (PHI - 1) if self.calibrate else r
        return self.values_[-1] * np.cumprod(steps)

    def _fitted_values(self) -> np.ndarray:
        # one-step-ahead using the mean growth rate (first point = itself)
        fitted = np.empty_like(self.values_)
        fitted[0] = self.values_[0]
        fitted[1:] = self.values_[:-1] * (1 + self.g_)
        return fitted


MODEL_CLASSES = (LinearTrend, ExponentialGrowth, FibonacciRatio)


# ---------------------------------------------------------------------------
# Evaluation, selection, bootstrap, planning
# ---------------------------------------------------------------------------
def evaluate_models(
    district: DistrictPopulation, last_train_year: int = 2021
) -> Tuple[Dict[str, Dict[str, float]], str]:
    """Train on years <= ``last_train_year``, score on the rest.

    Returns ``(table, best_model_name)`` where ``table[model][metric]`` holds
    MAE/RMSE/MAPE and the best model has the lowest RMSE.
    """
    train, test = district.split(last_train_year)
    table: Dict[str, Dict[str, float]] = {}
    for cls in MODEL_CLASSES:
        model = cls().fit(train.years, train.populations)
        pred = model.predict(len(test))
        table[model.name] = {
            "MAE": mae(test.populations, pred),
            "RMSE": rmse(test.populations, pred),
            "MAPE": mape(test.populations, pred),
        }
    best = min(table, key=lambda m: table[m]["RMSE"])
    return table, best


def model_by_name(name: str) -> Forecaster:
    """Instantiate a fresh model from its display name."""
    for cls in MODEL_CLASSES:
        if cls.name == name:
            return cls()
    raise KeyError(name)


def bootstrap_intervals(
    model: Forecaster,
    horizon: int,
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Residual-bootstrap prediction interval for ``horizon`` steps.

    Each resample builds ``y* = fitted + resampled residuals``, refits a copy
    of the model, forecasts, then adds one more resampled residual to represent
    observation noise. The (1-level)/2 and (1+level)/2 quantiles are returned.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    rng = np.random.default_rng(seed)
    resid = model.residuals()
    fitted = model.fitted_values()
    sims = np.empty((n_boot, horizon))
    for b in range(n_boot):
        y_star = fitted + rng.choice(resid, size=len(resid), replace=True)
        m = copy.deepcopy(model)
        m.fit(model.years_, y_star)
        sims[b] = m.predict(horizon) + rng.choice(resid, size=horizon, replace=True)
    alpha = (1 - level) / 2
    return np.quantile(sims, alpha, axis=0), np.quantile(sims, 1 - alpha, axis=0)


def classrooms_needed(
    population_thousands: float, school_age_share: float = 0.18, pupils_per_class: int = 53
) -> int:
    """Classrooms required (rounded up) for the primary-school-age population."""
    if population_thousands < 0 or not 0 <= school_age_share <= 1 or pupils_per_class <= 0:
        raise ValueError("invalid planning inputs")
    pupils = population_thousands * 1000 * school_age_share
    return math.ceil(pupils / pupils_per_class)


def load_districts() -> Dict[str, DistrictPopulation]:
    """Build all five ``DistrictPopulation`` objects from the bundled data."""
    return {n: DistrictPopulation(n, YEARS, v) for n, v in DISTRICT_DATA.items()}
