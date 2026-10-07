"""Loading, validating and generating budget data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ITEM_COLUMNS = [
    "item_id",
    "item_name",
    "department",
    "min_amount",
    "max_amount",
    "return_recession",
    "return_stable",
    "return_growth",
    "mandatory",
]
DEPARTMENT_COLUMNS = ["department", "min_share", "max_share"]


@dataclass
class BudgetProblem:
    """A budget allocation problem.

    Attributes:
        items: one row per budget item (see ``ITEM_COLUMNS``). Amounts are in
            million KZT; ``return_*`` is the expected effect per 1 unit spent
            under each scenario.
        departments: share limits of the total budget per department.
        total_budget: total amount available for allocation (million KZT).
    """

    items: pd.DataFrame
    departments: pd.DataFrame
    total_budget: float

    def __post_init__(self) -> None:
        validate(self)

    @property
    def n_items(self) -> int:
        return len(self.items)

    def with_budget(self, total_budget: float) -> "BudgetProblem":
        """Return a copy of the problem with another total budget."""
        return BudgetProblem(self.items.copy(), self.departments.copy(), total_budget)


def validate(problem: BudgetProblem) -> None:
    """Raise ``ValueError`` if the input data are inconsistent."""
    items, deps = problem.items, problem.departments

    missing = set(ITEM_COLUMNS) - set(items.columns)
    if missing:
        raise ValueError(f"items: missing columns {sorted(missing)}")
    missing = set(DEPARTMENT_COLUMNS) - set(deps.columns)
    if missing:
        raise ValueError(f"departments: missing columns {sorted(missing)}")

    if items["item_id"].duplicated().any():
        raise ValueError("items: item_id values must be unique")
    if (items["min_amount"] < 0).any():
        raise ValueError("items: min_amount must be non-negative")
    if (items["min_amount"] > items["max_amount"]).any():
        bad = items.loc[items["min_amount"] > items["max_amount"], "item_id"].tolist()
        raise ValueError(f"items: min_amount > max_amount for {bad}")

    unknown = set(items["department"]) - set(deps["department"])
    if unknown:
        raise ValueError(f"items: unknown departments {sorted(unknown)}")
    if ((deps["min_share"] < 0) | (deps["max_share"] > 1)).any():
        raise ValueError("departments: shares must lie in [0, 1]")
    if (deps["min_share"] > deps["max_share"]).any():
        raise ValueError("departments: min_share > max_share")
    if deps["min_share"].sum() > 1:
        raise ValueError("departments: sum of min_share exceeds 1")

    if problem.total_budget <= 0:
        raise ValueError("total_budget must be positive")


def load_problem(items_csv: str | Path, departments_csv: str | Path, total_budget: float) -> BudgetProblem:
    """Read a problem from two CSV files."""
    items = pd.read_csv(items_csv)
    items["mandatory"] = items["mandatory"].astype(bool)
    departments = pd.read_csv(departments_csv)
    return BudgetProblem(items, departments, float(total_budget))


DEFAULT_DEPARTMENTS = {
    # department: (min_share, max_share, [item names])
    "Production": (0.25, 0.45, ["Raw materials", "Equipment maintenance", "Energy", "Quality control"]),
    "Sales & Marketing": (0.08, 0.25, ["Digital advertising", "Trade shows", "Sales bonuses", "CRM licences"]),
    "R&D": (0.05, 0.20, ["Product prototyping", "Process automation", "Market research"]),
    "IT": (0.05, 0.15, ["Infrastructure", "Software licences", "Cybersecurity"]),
    "HR & Training": (0.03, 0.10, ["Staff training", "Recruitment"]),
    "Logistics": (0.05, 0.15, ["Warehousing", "Transport"]),
    "Administration": (0.03, 0.10, ["Office rent", "Legal & audit"]),
}

MANDATORY_ITEMS = {"Raw materials", "Energy", "Office rent", "Legal & audit", "Infrastructure"}

# Sensitivity of an item's effect to the economic cycle, in [-1, 1].
# Positive: pro-cyclical (pays off in growth, suffers in recession).
# Negative: counter-cyclical (cost-saving items that pay off most in a downturn).
CYCLICALITY = {
    "Digital advertising": 0.9,
    "Trade shows": 1.0,
    "Sales bonuses": 0.8,
    "Product prototyping": 0.7,
    "Market research": 0.4,
    "Recruitment": 0.8,
    "Warehousing": 0.5,
    "Transport": 0.3,
    "Equipment maintenance": -0.4,
    "Process automation": -0.8,
    "Energy": -0.3,
    "Software licences": -0.2,
    "Cybersecurity": -0.1,
    "Staff training": -0.2,
    "Quality control": -0.5,
    "CRM licences": 0.2,
}


def generate_synthetic(seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a reproducible synthetic dataset of an industrial enterprise.

    The numbers are illustrative and do not describe any real company.
    Each item gets a stable-economy return and a spread; the recession and
    growth returns move in opposite directions according to the item's
    cyclicality, so that no single scenario is the worst for every item.

    Returns:
        ``(items, departments)`` data frames.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for dep, (_, _, names) in DEFAULT_DEPARTMENTS.items():
        for name in names:
            mandatory = name in MANDATORY_ITEMS
            min_amount = round(float(rng.uniform(10, 60)), 1)
            max_amount = min_amount if mandatory else round(min_amount + float(rng.uniform(20, 120)), 1)
            stable = round(float(rng.uniform(1.05, 1.35)), 3)
            spread = float(rng.uniform(0.15, 0.40))
            c = CYCLICALITY.get(name, 0.0)
            rows.append(
                {
                    "item_id": f"I{len(rows) + 1:02d}",
                    "item_name": name,
                    "department": dep,
                    "min_amount": min_amount,
                    "max_amount": max_amount,
                    "return_recession": round(stable - c * spread, 3),
                    "return_stable": stable,
                    "return_growth": round(stable + c * spread, 3),
                    "mandatory": mandatory,
                }
            )
    items = pd.DataFrame(rows, columns=ITEM_COLUMNS)
    departments = pd.DataFrame(
        [(d, lo, hi) for d, (lo, hi, _) in DEFAULT_DEPARTMENTS.items()],
        columns=DEPARTMENT_COLUMNS,
    )
    return items, departments
