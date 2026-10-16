import numpy as np
import pytest

from src.microgrid import (
    CostModel, HybridMicroGrid, MicroGrid, SingularSystemError, describe,
    generate_demand_csv, load_demand_csv, parse_demand, prompt_demand,
)


def test_solve_day_known_answer():
    # x=2, y=3 -> D1 = 3*2+2*3 = 12, D2 = 4*2+3 = 11
    x, y = MicroGrid().solve_day(12, 11)
    assert (x, y) == pytest.approx((2, 3))


def test_determinant_and_condition():
    g = MicroGrid()
    assert g.determinant() == pytest.approx(-5)
    assert g.condition_number() >= 1


def test_loop_matches_vectorised(tmp_path):
    D = load_demand_csv(generate_demand_csv(tmp_path / "d.csv", 30, seed=1))
    g = MicroGrid()
    assert D.shape == (2, 30)
    assert np.allclose(g.solve_loop(D), g.solve_vectorised(D))


@pytest.mark.parametrize("bad", ["", "   ", "abc", "-3", "nan", "inf"])
def test_parse_demand_rejects_bad(bad):
    with pytest.raises(ValueError):
        parse_demand(bad)


def test_prompt_reprompts_until_valid():
    answers = iter(["", "x", "-1", "42.5"])
    msgs = []
    assert prompt_demand("D1? ", input_fn=lambda _: next(answers), print_fn=msgs.append) == 42.5
    assert len(msgs) == 3


def test_negative_demand_rejected_in_solve():
    with pytest.raises(ValueError):
        MicroGrid().solve_day(-1, 5)


def test_infeasible_day_flagged_and_nonnegative():
    # D2 far larger than D1 -> x>0, y<0 in the exact solution
    D = np.array([[10.0, 12.0], [40.0, 11.0]])
    rep = MicroGrid().handle_infeasible(D)
    assert rep.infeasible_days == [0]
    assert np.all(rep.handled >= 0)
    assert np.all(rep.raw[:, 1] == pytest.approx(rep.handled[:, 1]))


def test_singular_matrix_detected():
    g = MicroGrid([[1, 2], [2, 4]])
    with pytest.raises(SingularSystemError):
        g.solve_day(1, 2)


def test_hybrid_solution_and_dependent_row():
    h = HybridMicroGrid()
    s = np.array([2.0, 3.0, 1.0])
    d = h.A @ s
    assert h.solve_day(*d) == pytest.approx(s)
    dep = HybridMicroGrid([[3, 2, 1], [4, 1, 2], [7, 3, 3]])  # row3 = row1 + row2
    with pytest.raises(SingularSystemError):
        dep.solve_day(1, 1, 2)


def test_sensitivity_bounded_by_condition_number():
    g = MicroGrid()
    out = g.monte_carlo_sensitivity([60, 55], n_draws=500)
    assert np.all(out["rel_out"] <= g.condition_number() * out["rel_in"] + 1e-12)


def test_cost_model_and_describe():
    c = CostModel()
    assert c.daily_cost(np.array([[1.0, 2.0], [1.0, 0.0]])) == pytest.approx([600, 300])
    assert describe([1, 2, 3])["variance"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        describe([1])
