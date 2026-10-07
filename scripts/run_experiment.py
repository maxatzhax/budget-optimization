"""Run the full experiment: compare optimisation criteria and analyse budget sensitivity.

Outputs (in results/):
    allocation_comparison.csv   allocation per item under every criterion
    criteria_summary.csv        objective, worst/best-case effect per criterion
    budget_sensitivity.csv      objective and shadow price vs. total budget
    allocation_by_department.png
    budget_sensitivity.png

Usage:
    python scripts/run_experiment.py --budget 1000
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from budget_opt import budget_sensitivity, load_problem, solve_expected, solve_robust, solve_scenario  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--items", type=Path, default=ROOT / "data" / "sample" / "budget_items.csv")
    p.add_argument("--departments", type=Path, default=ROOT / "data" / "sample" / "departments.csv")
    p.add_argument("--budget", type=float, default=1000.0)
    p.add_argument("--out-dir", type=Path, default=ROOT / "results")
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    problem = load_problem(args.items, args.departments, args.budget)

    solutions = {
        "recession": solve_scenario(problem, "recession"),
        "stable": solve_scenario(problem, "stable"),
        "growth": solve_scenario(problem, "growth"),
        "expected": solve_expected(problem),
        "robust": solve_robust(problem),
    }

    # 1. allocation per item
    alloc = problem.items[["item_id", "item_name", "department"]].copy()
    for name, sol in solutions.items():
        alloc[name] = alloc["item_id"].map(sol.allocation).round(2)
    alloc.to_csv(args.out_dir / "allocation_comparison.csv", index=False)

    # 2. summary of criteria
    summary = pd.DataFrame(
        [
            {
                "criterion": name,
                "allocated": round(sol.total_allocated, 2),
                "effect_recession": round(sol.effects["recession"], 2),
                "effect_stable": round(sol.effects["stable"], 2),
                "effect_growth": round(sol.effects["growth"], 2),
                "worst_case": round(min(sol.effects.values()), 2),
                "budget_shadow_price": round(sol.budget_shadow_price, 4),
            }
            for name, sol in solutions.items()
        ]
    )
    summary.to_csv(args.out_dir / "criteria_summary.csv", index=False)
    print(summary.to_string(index=False))

    # 3. sensitivity to total budget
    min_needed = problem.items["min_amount"].sum()
    budgets = np.linspace(min_needed * 0.9, problem.items["max_amount"].sum() * 1.1, 40)
    sens = budget_sensitivity(problem, budgets, method="expected")
    sens.to_csv(args.out_dir / "budget_sensitivity.csv", index=False)

    # figures
    dep = pd.DataFrame({name: sol.by_department(problem) for name, sol in solutions.items()})
    ax = dep[["expected", "robust"]].plot.barh(figsize=(8, 4.5))
    ax.set_xlabel("Allocation, million KZT")
    ax.set_ylabel("")
    ax.set_title(f"Allocation by department (budget = {args.budget:,.0f})")
    plt.tight_layout()
    plt.savefig(args.out_dir / "allocation_by_department.png", dpi=150)
    plt.close()

    ok = sens[sens["status"] == "optimal"]
    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(ok["total_budget"], ok["objective"], marker="o", ms=3, label="Expected effect")
    ax1.set_xlabel("Total budget, million KZT")
    ax1.set_ylabel("Expected effect")
    ax2 = ax1.twinx()
    ax2.step(ok["total_budget"], ok["shadow_price"], where="post", color="tab:orange", label="Shadow price")
    ax2.set_ylabel("Shadow price of budget")
    ax1.set_title("Sensitivity of the optimal plan to the total budget")
    fig.legend(loc="upper left", bbox_to_anchor=(0.1, 0.88))
    fig.tight_layout()
    fig.savefig(args.out_dir / "budget_sensitivity.png", dpi=150)
    plt.close(fig)

    print(f"\nresults saved to {args.out_dir}")


if __name__ == "__main__":
    main()
