import numpy as np
import pandas as pd
import pytest

from budget_opt import BudgetProblem, budget_sensitivity, solve_expected, solve_robust, solve_scenario
from budget_opt.data import generate_synthetic


@pytest.fixture
def problem() -> BudgetProblem:
    items, deps = generate_synthetic(seed=42)
    return BudgetProblem(items, deps, total_budget=1000.0)


@pytest.fixture
def tiny() -> BudgetProblem:
    """Two items, one department: the optimum is obvious by hand."""
    items = pd.DataFrame(
        {
            "item_id": ["A", "B"],
            "item_name": ["a", "b"],
            "department": ["D", "D"],
            "min_amount": [0.0, 10.0],
            "max_amount": [60.0, 100.0],
            "return_recession": [1.0, 1.1],
            "return_stable": [2.0, 1.2],
            "return_growth": [3.0, 1.3],
            "mandatory": [False, False],
        }
    )
    deps = pd.DataFrame({"department": ["D"], "min_share": [0.0], "max_share": [1.0]})
    return BudgetProblem(items, deps, total_budget=100.0)


def test_tiny_stable_scenario_matches_hand_solution(tiny):
    # A has the higher stable-scenario return -> fill A up to 60, the rest (40) goes to B
    sol = solve_scenario(tiny, "stable")
    assert sol.allocation["A"] == pytest.approx(60)
    assert sol.allocation["B"] == pytest.approx(40)
    assert sol.objective == pytest.approx(60 * 2.0 + 40 * 1.2)
    # +1 unit of budget would go to B -> shadow price equals B's return
    assert sol.budget_shadow_price == pytest.approx(1.2)


def test_tiny_recession_prefers_b(tiny):
    sol = solve_scenario(tiny, "recession")
    assert sol.allocation["B"] == pytest.approx(100)
    assert sol.allocation["A"] == pytest.approx(0)


@pytest.mark.parametrize("solver", [solve_expected, solve_robust, lambda p: solve_scenario(p, "stable")])
def test_constraints_hold(problem, solver):
    sol = solver(problem)
    items, deps, B = problem.items, problem.departments, problem.total_budget
    x = sol.allocation.reindex(items["item_id"]).to_numpy()
    tol = 1e-6

    assert x.sum() <= B + tol
    assert np.all(x >= items["min_amount"].to_numpy() - tol)
    assert np.all(x <= items["max_amount"].to_numpy() + tol)

    mand = items["mandatory"].to_numpy()
    assert np.allclose(x[mand], items.loc[mand, "min_amount"].to_numpy())

    by_dep = sol.by_department(problem)
    for _, d in deps.iterrows():
        assert by_dep[d["department"]] >= d["min_share"] * x.sum() - tol
        assert by_dep[d["department"]] <= d["max_share"] * x.sum() + tol


def test_robust_has_best_worst_case(problem):
    robust = solve_robust(problem)
    worst = min(robust.effects.values())
    assert robust.objective == pytest.approx(worst)
    for other in (solve_expected(problem), solve_scenario(problem, "growth")):
        assert worst >= min(other.effects.values()) - 1e-6


def test_infeasible_budget_raises(problem):
    too_small = problem.items["min_amount"].sum() * 0.5
    with pytest.raises(RuntimeError):
        solve_expected(problem.with_budget(too_small))


def test_sensitivity_objective_non_decreasing(problem):
    budgets = np.linspace(900, 1600, 15)
    sens = budget_sensitivity(problem, budgets)
    ok = sens[sens["status"] == "optimal"]
    assert len(ok) > 0
    assert np.all(np.diff(ok["objective"].to_numpy()) >= -1e-6)


def test_validation_rejects_bad_bounds(tiny):
    items = tiny.items.copy()
    items.loc[0, "min_amount"] = 999
    with pytest.raises(ValueError):
        BudgetProblem(items, tiny.departments, 100.0)
