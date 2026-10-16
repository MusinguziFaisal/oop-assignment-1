import numpy as np
import pytest

from src.population import (
    DistrictPopulation, ExponentialGrowth, FibonacciRatio, LinearTrend,
    bootstrap_intervals, classrooms_needed, evaluate_models, fibonacci,
    load_districts, mape,
)


def test_valid_construction_len_repr():
    d = DistrictPopulation("X", [2020, 2021, 2022], [1, 2, 3])
    assert len(d) == 3
    assert "X" in repr(d)


@pytest.mark.parametrize(
    "years,pops",
    [([2020, 2021], [1]), ([2020, 2021], [1, -5]), ([], []), ([2021, 2020], [1, 2])],
)
def test_invalid_input_rejected(years, pops):
    with pytest.raises(ValueError):
        DistrictPopulation("bad", years, pops)


def test_cagr_known_value():
    d = DistrictPopulation("X", [2020, 2021, 2022], [100, 110, 121])
    assert d.cagr() == pytest.approx(0.10)
    assert d.yoy_growth() == pytest.approx([0.10, 0.10])


def test_variance_ddof_difference():
    d = DistrictPopulation("X", [1, 2, 3, 4], [2, 4, 4, 6])
    s = d.stats_statistics_module()["variance"]
    n = d.stats_numpy(ddof=0)["variance"]
    assert s == pytest.approx(n * 4 / 3)
    assert d.stats_numpy(ddof=1)["variance"] == pytest.approx(s)


def test_linear_recovers_perfect_line():
    years = np.arange(2015, 2022)
    m = LinearTrend().fit(years, 10 + 2 * (years - 2015))
    assert m.predict(2) == pytest.approx([24, 26])


def test_exponential_recovers_constant_growth():
    vals = 100 * 1.05 ** np.arange(7)
    m = ExponentialGrowth().fit(np.arange(7), vals)
    assert m.predict(1)[0] == pytest.approx(vals[-1] * 1.05)


def test_predict_before_fit_and_bad_horizon():
    with pytest.raises(RuntimeError):
        LinearTrend().predict(3)
    m = LinearTrend().fit([1, 2, 3], [1, 2, 3])
    with pytest.raises(ValueError):
        m.predict(0)


def test_fibonacci_sequence_and_calibrated_model_is_positive():
    assert fibonacci(8) == [1, 1, 2, 3, 5, 8, 13, 21]
    d = load_districts()["Kampala"]
    f = FibonacciRatio().fit(d.years, d.populations).predict(5)
    assert np.all(f > d.populations[-1])


def test_evaluate_models_returns_best():
    table, best = evaluate_models(load_districts()["Gulu"])
    assert best in table and set(table[best]) == {"MAE", "RMSE", "MAPE"}


def test_mape_zero_actual_raises():
    with pytest.raises(ValueError):
        mape([0, 1], [1, 1])


def test_bootstrap_interval_contains_point_forecast_and_is_ordered():
    d = load_districts()["Wakiso"]
    m = LinearTrend().fit(d.years, d.populations)
    lo, hi = bootstrap_intervals(m, 5, n_boot=300, seed=1)
    pt = m.predict(5)
    assert np.all(lo < hi) and np.all(lo <= pt) and np.all(pt <= hi)


def test_classrooms_edge_cases():
    assert classrooms_needed(0) == 0
    assert classrooms_needed(1000) == 3397  # 180000 / 53 = 3396.2, rounded up
