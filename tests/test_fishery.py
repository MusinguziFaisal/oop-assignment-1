import numpy as np
import pytest

from src.fishery import (
    FishStock, PriceModel, RiskAssessor, fibonacci_stock, weekly_revenue,
)


def test_fibonacci_baseline():
    f = fibonacci_stock(15)
    assert len(f) == 15 and f[-1] == 610


def test_stock_at_capacity_without_harvest_is_stable():
    stock, harvest = FishStock(n0=10_000, h=0).simulate(10)
    assert np.allclose(stock, 10_000) and np.allclose(harvest, 0)


def test_stock_converges_to_equilibrium():
    fs = FishStock(h=0.1)
    stock, _ = fs.simulate(300)
    assert stock[-1] == pytest.approx(fs.equilibrium(), rel=1e-6)
    assert fs.equilibrium() == pytest.approx(7500)


def test_msy_and_status():
    assert FishStock().msy() == pytest.approx(1000)
    assert FishStock(h=0.2).status() == "At MSY"
    assert FishStock(h=0.3).status() == "Overfishing"
    assert FishStock(h=0.05).status() == "Sustainable"
    assert FishStock(h=0.5).status() == "Collapse"


def test_h_above_r_collapses_and_stock_never_negative():
    stock, _ = FishStock(h=0.6).simulate(100)
    assert stock.min() >= 0 and stock[-1] < 1


@pytest.mark.parametrize("kwargs", [dict(h=-0.1), dict(h=1.5), dict(r=0), dict(K=-5), dict(n0=-1)])
def test_invalid_stock_params(kwargs):
    with pytest.raises(ValueError):
        FishStock(**kwargs)


def test_closed_season_zero_harvest():
    _, harvest = FishStock().simulate(52, closed_weeks=range(44, 52))
    assert np.all(harvest[44:] == 0) and np.all(harvest[:44] > 0)


def test_price_bounds_and_reproducibility():
    pm = PriceModel(seed=5)
    p1, p2 = pm.path(52), pm.path(52)
    assert np.array_equal(p1, p2) and p1[0] == 12_000
    assert p1.min() >= 9_000 and p1.max() <= 16_000
    with pytest.raises(ValueError):
        PriceModel(start=20_000)


def test_revenue_units():
    assert weekly_revenue(np.array([2.0]), np.array([10_000.0])) == pytest.approx([2e7])
    with pytest.raises(ValueError):
        weekly_revenue(np.ones(3), np.ones(2))


def test_cv_is_scale_free_but_variance_is_not():
    ra = RiskAssessor()
    base = [100, 110, 90, 105, 95]
    scaled = [x * 1000 for x in base]
    assert ra.describe(base)["cv"] == pytest.approx(ra.describe(scaled)["cv"])
    assert ra.describe(scaled)["variance"] == pytest.approx(ra.describe(base)["variance"] * 1e6)


def test_classify_thresholds_and_errors():
    ra = RiskAssessor(0.15, 0.30)
    assert [ra.classify(c) for c in (0.05, 0.2, 0.5)] == ["Low", "Moderate", "High"]
    with pytest.raises(ValueError):
        ra.classify(-1)
    with pytest.raises(ValueError):
        ra.describe([5])


def test_monte_carlo_var():
    ra = RiskAssessor()
    ann = ra.monte_carlo_annual(FishStock(), PriceModel(), n_paths=1000)
    assert ann.shape == (1000,)
    v = ra.value_at_risk(ann)
    assert v["var"] > 0 and v["quantile"] < v["mean"]
    with pytest.raises(ValueError):
        ra.monte_carlo_annual(FishStock(), PriceModel(), n_paths=10)
