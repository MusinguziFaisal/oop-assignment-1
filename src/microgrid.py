"""Mini-Project 2: Solar Micro-Grid Dispatch Planner.

Solves the daily dispatch problem ``A @ s = d`` for a health-centre micro-grid.
For the base grid ``s = (x, y)`` = (solar, battery) kWh and::

    3x + 2y = D1   (daytime load)
    4x +  y = D2   (critical-equipment load)

``HybridMicroGrid`` adds a diesel generator ``z`` and a third constraint.
"""
from __future__ import annotations

import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import linalg
from scipy.optimize import nnls

DEFAULT_MATRIX = np.array([[3.0, 2.0], [4.0, 1.0]])
COND_LIMIT = 1e12  # above this the system is treated as numerically singular


class SingularSystemError(ValueError):
    """Raised when the coefficient matrix cannot be solved reliably."""


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------
def parse_demand(text: str) -> float:
    """Convert text to a non-negative, finite float or raise ``ValueError``."""
    text = text.strip()
    if not text:
        raise ValueError("value is empty")
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"'{text}' is not a number") from None
    if not math.isfinite(value):
        raise ValueError("value must be finite")
    if value < 0:
        raise ValueError("value must not be negative")
    return value


def prompt_demand(
    prompt: str,
    input_fn: Callable[[str], str] = input,
    print_fn: Callable[[str], None] = print,
    max_attempts: Optional[int] = None,
) -> float:
    """Ask until a valid demand is entered (re-prompts on bad input).

    ``input_fn`` / ``print_fn`` are injectable so the loop can be unit-tested.
    ``max_attempts=None`` loops forever; otherwise ``ValueError`` after that many tries.
    """
    attempts = 0
    while max_attempts is None or attempts < max_attempts:
        attempts += 1
        try:
            return parse_demand(input_fn(prompt))
        except ValueError as exc:
            print_fn(f"Invalid input: {exc}. Please try again.")
    raise ValueError("too many invalid attempts")


# ---------------------------------------------------------------------------
# Data generation / loading
# ---------------------------------------------------------------------------
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def generate_demand_csv(path: str | Path, n_days: int = 30, seed: int = 2026) -> Path:
    """Write ``n_days`` of synthetic (D1, D2) demand to ``path`` (seeded).

    D1 follows a weekly pattern (busy clinic days Mon-Fri, quieter weekend) plus
    noise. D2 = D1 x ratio with a noisy ratio, so a few days fall outside the
    feasible region and exercise the infeasibility handling.
    """
    rng = np.random.default_rng(seed)
    weekly = np.array([1.00, 1.05, 1.02, 1.00, 0.95, 0.70, 0.60])  # Mon..Sun
    base = 80.0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["day", "weekday", "d1_kwh", "d2_kwh"])
        for i in range(n_days):
            d1 = base * weekly[i % 7] * (1 + rng.normal(0, 0.06))
            d2 = d1 * rng.normal(1.0, 0.25)
            w.writerow([i + 1, WEEKDAYS[i % 7], round(max(d1, 0), 2), round(max(d2, 0), 2)])
    return path


def load_demand_csv(path: str | Path) -> np.ndarray:
    """Load demand CSV into a (n_demands x n_days) array, validating each value."""
    path = Path(path)
    cols: List[List[float]] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        demand_cols = [c for c in (reader.fieldnames or []) if c.startswith("d") and c.endswith("_kwh")]
        if not demand_cols:
            raise ValueError("CSV has no demand columns like 'd1_kwh'")
        cols = [[] for _ in demand_cols]
        for row_no, row in enumerate(reader, start=2):
            for j, c in enumerate(demand_cols):
                try:
                    cols[j].append(parse_demand(row[c]))
                except ValueError as exc:
                    raise ValueError(f"{path.name} row {row_no}, column {c}: {exc}") from None
    if not cols[0]:
        raise ValueError("CSV contains no data rows")
    return np.array(cols, dtype=float)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class FeasibilityReport:
    """Outcome of handling infeasible days."""

    infeasible_days: List[int]  # 0-based indices where the exact solution had a negative entry
    raw: np.ndarray  # exact solutions (n x days)
    handled: np.ndarray  # non-negative dispatch actually used
    unmet: np.ndarray  # residual |A s - d| per constraint (n x days)

    @property
    def n_infeasible(self) -> int:
        return len(self.infeasible_days)


class MicroGrid:
    """Two-source micro-grid (solar ``x``, battery ``y``).

    Parameters
    ----------
    matrix:
        Square coefficient matrix. Defaults to ``[[3, 2], [4, 1]]``.
    """

    source_names: Tuple[str, ...] = ("solar", "battery")

    def __init__(self, matrix: Optional[Sequence[Sequence[float]]] = None) -> None:
        a = np.array(DEFAULT_MATRIX if matrix is None else matrix, dtype=float)
        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError("coefficient matrix must be square")
        self.A = a
        self.n = a.shape[0]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(A={self.A.tolist()})"

    # -- well-posedness ------------------------------------------------------
    def determinant(self) -> float:
        """det(A). Zero means no unique solution."""
        return float(np.linalg.det(self.A))

    def condition_number(self) -> float:
        """2-norm condition number of A (``np.linalg.cond``)."""
        return float(np.linalg.cond(self.A))

    def check_well_posed(self) -> Dict[str, float]:
        """Return det and cond; raise ``SingularSystemError`` if unusable."""
        det, cond = self.determinant(), self.condition_number()
        if abs(det) < 1e-12 or not np.isfinite(cond) or cond > COND_LIMIT:
            raise SingularSystemError(f"system is singular or ill-conditioned (det={det:.3g}, cond={cond:.3g})")
        return {"determinant": det, "condition_number": cond}

    # -- solving -------------------------------------------------------------
    def _validate(self, demands: Sequence[float]) -> np.ndarray:
        d = np.asarray(demands, dtype=float)
        if d.shape != (self.n,):
            raise ValueError(f"expected {self.n} demands, got {d.size}")
        if not np.all(np.isfinite(d)) or np.any(d < 0):
            raise ValueError("demands must be finite and non-negative")
        return d

    def solve_day(self, *demands: float) -> np.ndarray:
        """Exact solution for one day, e.g. ``solve_day(d1, d2)`` -> ``[x, y]``."""
        self.check_well_posed()
        return linalg.solve(self.A, self._validate(demands))

    def solve_loop(self, D: np.ndarray) -> np.ndarray:
        """Solve every column of ``D`` (n x days) with a Python loop."""
        return np.column_stack([self.solve_day(*D[:, j]) for j in range(D.shape[1])])

    def solve_vectorised(self, D: np.ndarray) -> np.ndarray:
        """Solve all days in one ``scipy.linalg.solve`` call (n x days right-hand side)."""
        D = np.asarray(D, dtype=float)
        if D.ndim != 2 or D.shape[0] != self.n:
            raise ValueError(f"D must have shape ({self.n}, days)")
        self.check_well_posed()
        return linalg.solve(self.A, D)

    # -- infeasibility -------------------------------------------------------
    def handle_infeasible(self, D: np.ndarray, tol: float = 1e-9) -> FeasibilityReport:
        """Flag days with a negative source and replace them by an NNLS dispatch.

        Strategy: ``scipy.optimize.nnls`` minimises ``||A s - d||`` subject to
        ``s >= 0``. Unlike clipping, it keeps the residual as small as possible
        and the report says how many kWh of each load stays unmet.
        """
        raw = self.solve_vectorised(D)
        bad = [int(j) for j in np.where((raw < -tol).any(axis=0))[0]]
        handled = raw.copy()
        for j in bad:
            handled[:, j], _ = nnls(self.A, np.asarray(D)[:, j])
        handled[np.abs(handled) < tol] = 0.0
        unmet = np.abs(self.A @ handled - D)
        return FeasibilityReport(bad, raw, handled, unmet)

    @staticmethod
    def clip_solution(raw: np.ndarray) -> np.ndarray:
        """Naive alternative: clip negatives to zero (loads then go unmet)."""
        return np.clip(raw, 0, None)

    # -- sensitivity ---------------------------------------------------------
    def monte_carlo_sensitivity(
        self, demands: Sequence[float], pct: float = 0.05, n_draws: int = 1000, seed: int = 7
    ) -> Dict[str, np.ndarray]:
        """Perturb every demand uniformly within +/- ``pct`` and re-solve.

        Returns the base solution, draws, the worst relative input change and the
        worst relative output change per draw (to compare with ``cond(A)``).
        """
        rng = np.random.default_rng(seed)
        d0 = self._validate(demands)
        base = self.solve_day(*d0)
        factors = 1 + rng.uniform(-pct, pct, size=(self.n, n_draws))
        D = d0[:, None] * factors
        sols = linalg.solve(self.A, D)
        rel_in = np.linalg.norm(D - d0[:, None], axis=0) / np.linalg.norm(d0)
        rel_out = np.linalg.norm(sols - base[:, None], axis=0) / np.linalg.norm(base)
        return {"base": base, "solutions": sols, "rel_in": rel_in, "rel_out": rel_out}


class HybridMicroGrid(MicroGrid):
    """Micro-grid with a diesel generator ``z`` and a third constraint.

    Default third equation: ``x + y + z = D3`` (total energy dispatched balances
    the day's total requirement). Constraint 1 and 2 gain the diesel coefficients
    (1 and 2).
    """

    source_names = ("solar", "battery", "diesel")
    DEFAULT = np.array([[3.0, 2.0, 1.0], [4.0, 1.0, 2.0], [1.0, 1.0, 1.0]])

    def __init__(self, matrix: Optional[Sequence[Sequence[float]]] = None) -> None:
        super().__init__(self.DEFAULT if matrix is None else matrix)
        if self.n != 3:
            raise ValueError("HybridMicroGrid needs a 3x3 matrix")

    def solve_day(self, d1: float, d2: float, d3: float) -> np.ndarray:  # type: ignore[override]
        """Exact solution ``[x, y, z]`` for the three demands."""
        return super().solve_day(d1, d2, d3)


# ---------------------------------------------------------------------------
# Statistics and cost
# ---------------------------------------------------------------------------
def describe(values: Sequence[float]) -> Dict[str, float]:
    """Mean, sample variance, sample std and coefficient of variation (statistics module)."""
    v = list(map(float, values))
    if len(v) < 2:
        raise ValueError("need at least two values")
    mean = statistics.mean(v)
    sd = statistics.stdev(v)
    return {"mean": mean, "variance": statistics.variance(v), "std": sd,
            "cv": sd / mean if mean else float("nan")}


@dataclass(frozen=True)
class CostModel:
    """Energy prices in UGX per kWh (illustrative)."""

    solar_price: float = 150.0
    battery_price: float = 450.0
    diesel_price: float = 1800.0

    def __post_init__(self) -> None:
        if min(self.solar_price, self.battery_price, self.diesel_price) < 0:
            raise ValueError("prices must be non-negative")

    def daily_cost(self, dispatch: np.ndarray) -> np.ndarray:
        """Cost per day for a (n_sources x days) dispatch array (solar, battery[, diesel])."""
        prices = np.array([self.solar_price, self.battery_price, self.diesel_price])[: dispatch.shape[0]]
        return prices @ dispatch

    def monthly_cost(self, dispatch: np.ndarray) -> float:
        """Total cost over all days."""
        return float(self.daily_cost(dispatch).sum())
