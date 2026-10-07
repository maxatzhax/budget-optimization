"""Linear programming models of enterprise budget allocation.

Decision variables: ``x_i`` — amount allocated to budget item ``i``.

Common constraints (see ``docs/model.md`` for the full formulation):

* total budget:          sum_i x_i <= B
* item bounds:           l_i <= x_i <= u_i   (l_i = u_i for mandatory items)
* department shares:     a_d * X <= sum_{i in d} x_i <= b_d * X,  X = sum_i x_i

Models:

* ``solve_scenario``  — maximise the effect under one scenario of returns;
* ``solve_expected``  — maximise the probability-weighted expected effect;
* ``solve_robust``    — maximise the worst-case effect over all scenarios
  (max-min / Wald criterion), linearised with an auxiliary variable ``t``.

All models are solved with the HiGHS solver via ``scipy.optimize.linprog``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from budget_opt.data import BudgetProblem

SCENARIOS = ("recession", "stable", "growth")
DEFAULT_WEIGHTS = {"recession": 0.30, "stable": 0.50, "growth": 0.20}


@dataclass
class Solution:
    """Result of an optimisation run."""

    method: str
    status: str
    objective: float
    allocation: pd.Series
    budget_shadow_price: float
    effects: dict[str, float] = field(default_factory=dict)

    @property
    def total_allocated(self) -> float:
        return float(self.allocation.sum())

    def by_department(self, problem: BudgetProblem) -> pd.Series:
        dep = problem.items.set_index("item_id")["department"]
        return self.allocation.groupby(dep).sum()

    def to_frame(self, problem: BudgetProblem) -> pd.DataFrame:
        df = problem.items[["item_id", "item_name", "department", "min_amount", "max_amount"]].copy()
        df["allocation"] = df["item_id"].map(self.allocation).round(3)
        return df


def _returns(problem: BudgetProblem, scenario: str) -> np.ndarray:
    if scenario not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario!r}; expected one of {SCENARIOS}")
    return problem.items[f"return_{scenario}"].to_numpy(dtype=float)


def _common_constraints(problem: BudgetProblem, n_extra: int = 0):
    """Build ``A_ub``, ``b_ub`` and bounds shared by all models.

    Row 0 of ``A_ub`` is always the total budget constraint.
    ``n_extra`` additional zero columns are appended for auxiliary variables.
    """
    items, deps, B = problem.items, problem.departments, problem.total_budget
    n = len(items)
    rows, rhs = [], []

    # total budget
    rows.append(np.r_[np.ones(n), np.zeros(n_extra)])
    rhs.append(B)

    # department share limits, relative to the total amount actually allocated:
    #   a_d * sum_i x_i <= sum_{i in d} x_i <= b_d * sum_i x_i
    ones = np.ones(n)
    for _, d in deps.iterrows():
        mask = (items["department"] == d["department"]).to_numpy(dtype=float)
        rows.append(np.r_[-(mask - d["min_share"] * ones), np.zeros(n_extra)])
        rhs.append(0.0)
        rows.append(np.r_[mask - d["max_share"] * ones, np.zeros(n_extra)])
        rhs.append(0.0)

    lower = items["min_amount"].to_numpy(dtype=float)
    upper = np.where(items["mandatory"], lower, items["max_amount"].to_numpy(dtype=float))
    bounds = list(zip(lower, upper))
    return np.vstack(rows), np.array(rhs), bounds


def _effects(problem: BudgetProblem, x: np.ndarray) -> dict[str, float]:
    return {s: float(_returns(problem, s) @ x) for s in SCENARIOS}


def _solve(problem: BudgetProblem, c, A_ub, b_ub, bounds, method: str, objective_sign: float = -1.0) -> Solution:
    n = problem.n_items
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if res.status != 0:
        raise RuntimeError(f"{method}: optimisation failed — {res.message}")
    x = res.x[:n]
    # linprog minimises; we maximise, so flip the objective and the dual of the budget row
    shadow = float(objective_sign * res.ineqlin.marginals[0])
    return Solution(
        method=method,
        status=res.message,
        objective=float(objective_sign * res.fun),
        allocation=pd.Series(x, index=problem.items["item_id"].to_numpy(), name="allocation"),
        budget_shadow_price=shadow,
        effects=_effects(problem, x),
    )


def solve_scenario(problem: BudgetProblem, scenario: str = "stable") -> Solution:
    """Maximise the total effect under a single scenario of returns."""
    r = _returns(problem, scenario)
    A_ub, b_ub, bounds = _common_constraints(problem)
    return _solve(problem, -r, A_ub, b_ub, bounds, method=f"scenario:{scenario}")


def solve_expected(problem: BudgetProblem, weights: dict[str, float] | None = None) -> Solution:
    """Maximise the probability-weighted expected effect (Bayes criterion)."""
    weights = weights or DEFAULT_WEIGHTS
    if abs(sum(weights.values()) - 1) > 1e-9:
        raise ValueError("scenario weights must sum to 1")
    r = sum(w * _returns(problem, s) for s, w in weights.items())
    A_ub, b_ub, bounds = _common_constraints(problem)
    return _solve(problem, -r, A_ub, b_ub, bounds, method="expected")


def solve_robust(problem: BudgetProblem) -> Solution:
    """Maximise the worst-case effect over all scenarios (Wald max-min criterion).

    max t  s.t.  t <= r^s · x  for every scenario s, plus common constraints.
    """
    n = problem.n_items
    A_ub, b_ub, bounds = _common_constraints(problem, n_extra=1)
    scen_rows = [np.r_[-_returns(problem, s), 1.0] for s in SCENARIOS]  # t - r^s x <= 0
    A_ub = np.vstack([A_ub, *scen_rows])
    b_ub = np.r_[b_ub, np.zeros(len(SCENARIOS))]
    bounds = bounds + [(None, None)]
    c = np.r_[np.zeros(n), -1.0]
    return _solve(problem, c, A_ub, b_ub, bounds, method="robust")


def budget_sensitivity(problem: BudgetProblem, budgets, method: str = "expected") -> pd.DataFrame:
    """Re-solve the model for a range of total budgets.

    Returns a frame with the objective value, the amount actually allocated and
    the shadow price of the budget constraint (marginal effect of +1 unit of
    budget). Infeasible budgets are reported with ``status = "infeasible"``.
    """
    solver = {"expected": solve_expected, "robust": solve_robust, "stable": solve_scenario}[method]
    rows = []
    for B in budgets:
        try:
            sol = solver(problem.with_budget(float(B)))
            rows.append(
                {
                    "total_budget": float(B),
                    "status": "optimal",
                    "objective": sol.objective,
                    "allocated": sol.total_allocated,
                    "shadow_price": sol.budget_shadow_price,
                }
            )
        except (RuntimeError, ValueError):
            rows.append({"total_budget": float(B), "status": "infeasible"})
    return pd.DataFrame(rows)
