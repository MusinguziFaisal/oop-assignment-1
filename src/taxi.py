"""Mini-Project 5: Taxi Route Revenue, Pricing & Fleet Planner."""
from __future__ import annotations

import math
import statistics
from abc import ABC, abstractmethod
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy import linalg

from src.population import mae  # reuse the error metric


# ---------------------------------------------------------------------------
# Data (daily passengers over 10 days). Ntinda is from the original question;
# Entebbe and Mukono are illustrative.
# ---------------------------------------------------------------------------
ROUTE_DATA: Dict[str, Tuple[List[int], float]] = {
    "Kampala-Ntinda": ([35, 40, 42, 50, 55, 60, 48, 52, 47, 45], 2000),
    "Kampala-Entebbe": ([60, 58, 65, 70, 72, 80, 75, 68, 66, 64], 5000),
    "Kampala-Mukono": ([45, 47, 50, 49, 55, 62, 58, 53, 51, 50], 3000),
}


class Route:
    """A taxi route: daily passenger counts and a fixed fare (UGX per passenger)."""

    def __init__(self, name: str, passengers: Sequence[float], fare: float) -> None:
        p = np.asarray(passengers, dtype=float)
        if p.ndim != 1 or len(p) == 0:
            raise ValueError("passengers must be a non-empty 1-D sequence")
        if not np.all(np.isfinite(p)) or np.any(p < 0):
            raise ValueError("passenger counts must be finite and non-negative")
        if fare <= 0:
            raise ValueError("fare must be positive")
        self.name, self.passengers, self.fare = name, p, float(fare)

    def __repr__(self) -> str:
        return f"Route({self.name!r}, days={len(self)}, fare=UGX {self.fare:,.0f})"

    def __len__(self) -> int:
        return len(self.passengers)

    def daily_revenue(self) -> np.ndarray:
        """Revenue per day (UGX)."""
        return self.passengers * self.fare

    def total_revenue(self) -> float:
        return float(self.daily_revenue().sum())

    def describe(self) -> Dict[str, float]:
        """Mean, sample variance and std of daily passengers (``statistics`` module)."""
        if len(self) < 2:
            raise ValueError("need at least two days")
        v = self.passengers.tolist()
        return {"mean": statistics.mean(v), "variance": statistics.variance(v), "std": statistics.stdev(v)}


class SupplyDemand:
    """Linear market: ``Qd = a - b P`` and ``Qs = c + d P``.

    Written as a 2x2 system in (Q, P): ``Q + b P = a`` and ``Q - d P = c``.
    """

    def __init__(self, a: float = 120.0, b: float = 0.02, c: float = 10.0, d: float = 0.03) -> None:
        if b <= 0 or d <= 0:
            raise ValueError("demand slope b and supply slope d must be positive")
        self.a, self.b, self.c, self.d = a, b, c, d

    def demand(self, price: float) -> float:
        return self.a - self.b * price

    def supply(self, price: float) -> float:
        return self.c + self.d * price

    def equilibrium(self) -> Tuple[float, float]:
        """Return ``(Q*, P*)`` solved with ``scipy.linalg.solve``."""
        A = np.array([[1.0, self.b], [1.0, -self.d]])
        rhs = np.array([self.a, self.c])
        q, p = linalg.solve(A, rhs)
        return float(q), float(p)

    def excess_demand(self, price: float) -> float:
        """Qd - Qs at ``price`` (positive = shortage, negative = surplus)."""
        return self.demand(price) - self.supply(price)


# ---------------------------------------------------------------------------
# Forecasters
# ---------------------------------------------------------------------------
class Forecaster(ABC):
    """Base class: ``fit(history)`` then ``predict(horizon)`` (non-negative counts)."""

    name = "Forecaster"
    min_history = 1

    def __init__(self) -> None:
        self.history = np.array([])

    def fit(self, history: Sequence[float]) -> "Forecaster":
        h = np.asarray(history, dtype=float)
        if len(h) < self.min_history:
            raise ValueError(f"{self.name} needs at least {self.min_history} observations")
        if np.any(h < 0) or not np.all(np.isfinite(h)):
            raise ValueError("history must be finite and non-negative")
        self.history = h
        self._fit()
        return self

    def predict(self, horizon: int = 1) -> np.ndarray:
        if len(self.history) == 0:
            raise RuntimeError("call fit() first")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        return np.clip(self._predict(int(horizon)), 0, None)

    def _fit(self) -> None:  # optional hook
        pass

    @abstractmethod
    def _predict(self, horizon: int) -> np.ndarray: ...


class MovingAverage(Forecaster):
    """Mean of the last ``window`` observations (recursive for horizon > 1)."""

    min_history = 1

    def __init__(self, window: int = 3) -> None:
        super().__init__()
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self.name = f"{window}-day moving average"
        self.min_history = window

    def _predict(self, horizon: int) -> np.ndarray:
        buf = list(self.history)
        out = []
        for _ in range(horizon):
            nxt = float(np.mean(buf[-self.window:]))
            out.append(nxt)
            buf.append(nxt)
        return np.array(out)


class ExponentialSmoothing(Forecaster):
    """Simple exponential smoothing with smoothing factor ``alpha`` in (0, 1]."""

    def __init__(self, alpha: float = 0.5) -> None:
        super().__init__()
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.name = f"SES (alpha={alpha:.2f})"

    def _fit(self) -> None:
        level = self.history[0]
        for y in self.history[1:]:
            level = self.alpha * y + (1 - self.alpha) * level
        self.level_ = float(level)

    def _predict(self, horizon: int) -> np.ndarray:
        return np.full(horizon, self.level_)


class LinearTrend(Forecaster):
    """Straight line through all history (``np.polyfit``), extrapolated."""

    name = "Linear trend"
    min_history = 2

    def _fit(self) -> None:
        x = np.arange(len(self.history))
        self.slope_, self.intercept_ = np.polyfit(x, self.history, 1)

    def _predict(self, horizon: int) -> np.ndarray:
        x = len(self.history) + np.arange(horizon)
        return self.slope_ * x + self.intercept_


class SeasonalNaive(Forecaster):
    """Forecast = the value one season (``period`` days) earlier."""

    def __init__(self, period: int = 7) -> None:
        super().__init__()
        if period < 1:
            raise ValueError("period must be >= 1")
        self.period = period
        self.name = f"Seasonal naive (period {period})"
        self.min_history = period

    def _predict(self, horizon: int) -> np.ndarray:
        n = len(self.history)
        return np.array([self.history[n - self.period + (h % self.period)] for h in range(horizon)])


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def rolling_origin_forecasts(
    factory: Callable[[], Forecaster], series: Sequence[float], first_target_day: int = 4
) -> Tuple[np.ndarray, np.ndarray]:
    """One-step-ahead walk-forward forecasts.

    For each target day ``d`` (1-based, from ``first_target_day`` to the last day)
    fit on days ``1..d-1`` and predict day ``d``. Returns ``(actual, forecast)``.
    """
    y = np.asarray(series, dtype=float)
    if first_target_day < 2 or first_target_day > len(y):
        raise ValueError("first_target_day must be between 2 and len(series)")
    actual, forecast = [], []
    for d in range(first_target_day, len(y) + 1):
        model = factory().fit(y[: d - 1])
        forecast.append(float(model.predict(1)[0]))
        actual.append(y[d - 1])
    return np.array(actual), np.array(forecast)


def rolling_mae(factory: Callable[[], Forecaster], series: Sequence[float], first_target_day: int = 4) -> float:
    """Walk-forward MAE."""
    a, f = rolling_origin_forecasts(factory, series, first_target_day)
    return mae(a, f)


def grid_search_alpha(
    series: Sequence[float], first_target_day: int = 4, alphas: Sequence[float] | None = None
) -> Tuple[float, Dict[float, float]]:
    """Pick the alpha with the lowest walk-forward MAE. Returns ``(best, {alpha: mae})``."""
    grid = np.round(np.arange(0.05, 1.0001, 0.05), 2) if alphas is None else np.asarray(alphas)
    scores = {float(a): rolling_mae(lambda a=a: ExponentialSmoothing(float(a)), series, first_target_day) for a in grid}
    return min(scores, key=scores.get), scores


def vehicles_needed(
    passengers: float, trips_per_day: int = 8, seats: int = 14, buffer: float = 0.15
) -> int:
    """Vehicles to cover ``passengers`` per day with a safety ``buffer``, rounded up (min 1 if any demand)."""
    if passengers < 0 or trips_per_day <= 0 or seats <= 0 or buffer < 0:
        raise ValueError("invalid fleet inputs")
    if passengers == 0:
        return 0
    return max(1, math.ceil(passengers * (1 + buffer) / (trips_per_day * seats)))


def simulate_weekly_demand(days: int = 60, base: float = 46.0, noise_sd: float = 4.0, seed: int = 2026) -> np.ndarray:
    """Synthetic daily passengers with a weekly pattern (day 1 = Monday).

    Busier on Fridays, quieter on Sundays, plus seeded Gaussian noise.
    """
    if days < 1:
        raise ValueError("days must be >= 1")
    rng = np.random.default_rng(seed)
    weekly = np.array([1.00, 0.98, 1.00, 1.05, 1.30, 1.10, 0.70])  # Mon..Sun
    mult = weekly[np.arange(days) % 7]
    return np.clip(np.round(base * mult + rng.normal(0, noise_sd, days)), 0, None)


def load_routes() -> Dict[str, Route]:
    """Build the three ``Route`` objects."""
    return {n: Route(n, p, f) for n, (p, f) in ROUTE_DATA.items()}
