import numpy as np
import pytest

from src.taxi import (
    ExponentialSmoothing, LinearTrend, MovingAverage, Route, SeasonalNaive,
    SupplyDemand, grid_search_alpha, load_routes, rolling_mae,
    rolling_origin_forecasts, simulate_weekly_demand, vehicles_needed,
)


def test_route_revenue_and_stats():
    r = Route("R", [10, 20, 30], 2000)
    assert r.daily_revenue() == pytest.approx([20_000, 40_000, 60_000])
    assert r.total_revenue() == 120_000
    assert r.describe()["variance"] == pytest.approx(100)
    assert len(r) == 3


@pytest.mark.parametrize("p,f", [([], 100), ([-1, 2], 100), ([1, 2], 0)])
def test_route_validation(p, f):
    with pytest.raises(ValueError):
        Route("bad", p, f)


def test_equilibrium_matches_hand_calculation():
    q, p = SupplyDemand().equilibrium()
    assert p == pytest.approx(2200) and q == pytest.approx(76)
    sd = SupplyDemand()
    assert sd.demand(2000) == pytest.approx(80) and sd.supply(2000) == pytest.approx(70)
    assert sd.excess_demand(2000) == pytest.approx(10)
    assert sd.excess_demand(2200) == pytest.approx(0, abs=1e-9)


def test_moving_average_and_edge_cases():
    assert MovingAverage(3).fit([1, 2, 3, 4]).predict(1)[0] == pytest.approx(3.0)
    with pytest.raises(ValueError):
        MovingAverage(3).fit([1, 2])  # not enough history
    with pytest.raises(RuntimeError):
        MovingAverage(3).predict(1)


def test_ses_alpha_one_is_last_value_and_validation():
    assert ExponentialSmoothing(1.0).fit([5, 9, 7]).predict(1)[0] == 7
    with pytest.raises(ValueError):
        ExponentialSmoothing(0)


def test_linear_trend_extrapolates_line():
    assert LinearTrend().fit([2, 4, 6, 8]).predict(2) == pytest.approx([10, 12])


def test_predictions_never_negative():
    assert LinearTrend().fit([6, 4, 2]).predict(5).min() >= 0


def test_seasonal_naive():
    hist = [1, 2, 3, 4, 5, 6, 7, 10]
    assert SeasonalNaive(7).fit(hist).predict(2) == pytest.approx([2, 3])


def test_rolling_origin_has_no_lookahead_and_right_length():
    series = np.arange(1.0, 11.0)
    actual, fc = rolling_origin_forecasts(lambda: MovingAverage(3), series, 4)
    assert len(actual) == 7 and actual[0] == 4
    assert fc[0] == pytest.approx(2.0)  # mean of days 1-3 only
    with pytest.raises(ValueError):
        rolling_origin_forecasts(lambda: MovingAverage(3), series, 11)


def test_grid_search_alpha_returns_best_in_grid():
    series = [35, 40, 42, 50, 55, 60, 48, 52, 47, 45]
    best, scores = grid_search_alpha(series)
    assert scores[best] == min(scores.values()) and 0 < best <= 1


def test_vehicle_planning_edge_cases():
    assert vehicles_needed(0) == 0
    assert vehicles_needed(1) == 1
    assert vehicles_needed(112) == 2          # 112 * 1.15 / 112 = 1.15 -> 2
    assert vehicles_needed(112, buffer=0) == 1
    with pytest.raises(ValueError):
        vehicles_needed(-5)


def test_seasonal_naive_beats_moving_average_on_weekly_data():
    y = simulate_weekly_demand(60, seed=2026)
    assert np.array_equal(y, simulate_weekly_demand(60, seed=2026))  # reproducible
    assert rolling_mae(lambda: SeasonalNaive(7), y, 15) < rolling_mae(lambda: MovingAverage(3), y, 15)


def test_load_routes():
    routes = load_routes()
    assert set(routes) == {"Kampala-Ntinda", "Kampala-Entebbe", "Kampala-Mukono"}
