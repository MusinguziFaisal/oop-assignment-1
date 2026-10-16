"""Mini-Project 3: Lake Victoria Fish Stock & Export Risk Model.

``FishStock`` simulates discrete logistic growth with proportional harvesting,
``PriceModel`` a bounded random-walk price, and ``RiskAssessor`` turns weekly
revenue into a coefficient-of-variation risk class and a Monte Carlo
Value-at-Risk.
"""
from __future__ import annotations

import statistics
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from src.population import fibonacci  # reuse: no duplicated Fibonacci logic

WEEKS_PER_YEAR = 52


def fibonacci_stock(n: int = 15) -> List[int]:
    """Baseline 'stock' of the first ``n`` Fibonacci numbers (unbounded growth)."""
    return fibonacci(n)


class FishStock:
    """Discrete logistic growth with harvesting.

    ``N(t+1) = N(t) + r N(t) (1 - N(t)/K) - h N(t)``; the week-t harvest is
    ``h N(t)`` tonnes.

    Parameters
    ----------
    r: intrinsic weekly growth rate (> 0).
    K: carrying capacity in tonnes (> 0).
    n0: initial stock in tonnes (>= 0).
    h: weekly harvest fraction, 0 <= h <= 1.
    """

    def __init__(self, r: float = 0.4, K: float = 10_000.0, n0: float = 4_000.0, h: float = 0.1) -> None:
        if r <= 0 or K <= 0:
            raise ValueError("r and K must be positive")
        if n0 < 0:
            raise ValueError("initial stock must be non-negative")
        if not 0 <= h <= 1:
            raise ValueError("harvest rate h must be between 0 and 1")
        self.r, self.K, self.n0, self.h = float(r), float(K), float(n0), float(h)

    def __repr__(self) -> str:
        return f"FishStock(r={self.r}, K={self.K}, n0={self.n0}, h={self.h})"

    def simulate(
        self, weeks: int = WEEKS_PER_YEAR, closed_weeks: Optional[Iterable[int]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the model.

        Returns ``(stock, harvest)`` where ``stock`` has ``weeks + 1`` values
        (week 0 ... week ``weeks``) and ``harvest`` has ``weeks`` values (tonnes).
        ``closed_weeks`` holds week-of-year indices (0-51) with no harvesting.
        """
        if weeks < 1:
            raise ValueError("weeks must be >= 1")
        closed = set(int(w) % WEEKS_PER_YEAR for w in (closed_weeks or []))
        stock = np.empty(weeks + 1)
        harvest = np.zeros(weeks)
        stock[0] = self.n0
        for t in range(weeks):
            n = stock[t]
            h = 0.0 if (t % WEEKS_PER_YEAR) in closed else self.h
            harvest[t] = h * n
            stock[t + 1] = max(n + self.r * n * (1 - n / self.K) - harvest[t], 0.0)
        return stock, harvest

    def equilibrium(self) -> float:
        """Steady-state stock ``K (1 - h/r)`` (0 if ``h >= r``: collapse)."""
        return max(self.K * (1 - self.h / self.r), 0.0)

    def sustainable_yield(self) -> float:
        """Weekly harvest (tonnes) at the equilibrium stock."""
        return self.h * self.equilibrium()

    def msy(self) -> float:
        """Maximum sustainable yield ``rK/4`` tonnes per week (reached at h = r/2)."""
        return self.r * self.K / 4

    def status(self, tol: float = 1e-9) -> str:
        """Biological status from the harvest rate alone.

        ``Collapse`` if h >= r, ``Overfishing`` if h > r/2 (equilibrium stock below
        K/2, so yield is below MSY and stock is depleted), ``At MSY`` if h = r/2,
        otherwise ``Sustainable``.
        """
        if self.h >= self.r:
            return "Collapse"
        if abs(self.h - self.r / 2) <= tol:
            return "At MSY"
        return "Overfishing" if self.h > self.r / 2 else "Sustainable"


class PriceModel:
    """Weekly price (UGX/kg) as a seeded, bounded Gaussian random walk."""

    def __init__(self, start: float = 12_000.0, sigma: float = 350.0,
                 low: float = 9_000.0, high: float = 16_000.0, seed: int = 42) -> None:
        if not low <= start <= high:
            raise ValueError("start price must lie within [low, high]")
        if sigma < 0:
            raise ValueError("sigma must be non-negative")
        self.start, self.sigma, self.low, self.high, self.seed = start, sigma, low, high, seed

    def paths(self, weeks: int, n_paths: int = 1, seed: Optional[int] = None) -> np.ndarray:
        """Simulate ``n_paths`` price paths of length ``weeks`` (shape ``n_paths x weeks``)."""
        if weeks < 1 or n_paths < 1:
            raise ValueError("weeks and n_paths must be >= 1")
        rng = np.random.default_rng(self.seed if seed is None else seed)
        steps = rng.normal(0.0, self.sigma, size=(n_paths, weeks))
        out = np.empty((n_paths, weeks))
        out[:, 0] = self.start
        for t in range(1, weeks):
            out[:, t] = np.clip(out[:, t - 1] + steps[:, t], self.low, self.high)
        return out

    def path(self, weeks: int) -> np.ndarray:
        """A single reproducible price path."""
        return self.paths(weeks, 1)[0]


def weekly_revenue(harvest_tonnes: np.ndarray, price_ugx_per_kg: np.ndarray) -> np.ndarray:
    """Revenue in UGX = harvest (kg) x price (UGX/kg)."""
    h, p = np.asarray(harvest_tonnes, float), np.asarray(price_ugx_per_kg, float)
    if h.shape != p.shape:
        raise ValueError("harvest and price must have the same shape")
    return h * 1000.0 * p


class RiskAssessor:
    """Risk classification by coefficient of variation (CV) plus Monte Carlo VaR.

    Rule (CV = std / mean): ``< low`` -> Low, ``< high`` -> Moderate, else High.
    Defaults 0.15 / 0.30 follow the common rule of thumb that revenue streams
    with under ~15 % relative dispersion are stable and over ~30 % are volatile.
    CV is unit-free, so unlike a raw variance cut-off it does not change when the
    currency or the scale of the operation changes.
    """

    def __init__(self, low: float = 0.15, high: float = 0.30) -> None:
        if not 0 < low < high:
            raise ValueError("need 0 < low < high")
        self.low, self.high = low, high

    @staticmethod
    def describe(values: Iterable[float]) -> Dict[str, float]:
        """Mean, median, variance, std and CV using the ``statistics`` module."""
        v = [float(x) for x in values]
        if len(v) < 2:
            raise ValueError("need at least two values")
        mean = statistics.mean(v)
        if mean == 0:
            raise ValueError("mean is zero: CV undefined")
        sd = statistics.stdev(v)
        return {"mean": mean, "median": statistics.median(v), "variance": statistics.variance(v),
                "std": sd, "cv": sd / mean}

    def classify(self, cv: float) -> str:
        """Map a CV to Low / Moderate / High."""
        if cv < 0:
            raise ValueError("CV cannot be negative")
        return "Low" if cv < self.low else ("Moderate" if cv < self.high else "High")

    def monte_carlo_annual(self, stock: FishStock, prices: PriceModel, n_paths: int = 2000,
                           weeks: int = WEEKS_PER_YEAR, seed: int = 123,
                           closed_weeks: Optional[Iterable[int]] = None) -> np.ndarray:
        """Annual revenue (UGX) for each simulated price path."""
        if n_paths < 1000:
            raise ValueError("use at least 1000 paths")
        _, harvest = stock.simulate(weeks, closed_weeks)
        paths = prices.paths(weeks, n_paths, seed=seed)
        return (harvest * 1000.0 * paths).sum(axis=1)

    @staticmethod
    def value_at_risk(annual: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
        """5 % VaR: expected revenue minus the alpha-quantile (the shortfall)."""
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        a = np.asarray(annual, float)
        q = float(np.quantile(a, alpha))
        return {"mean": float(a.mean()), "quantile": q, "var": float(a.mean() - q)}
