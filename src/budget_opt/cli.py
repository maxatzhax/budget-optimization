"""Command-line interface: ``python -m budget_opt.cli --help``."""

from __future__ import annotations

import argparse
from pathlib import Path

from budget_opt.data import load_problem
from budget_opt.model import solve_expected, solve_robust, solve_scenario


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Optimal allocation of an enterprise budget (LP).")
    p.add_argument("--items", default="data/sample/budget_items.csv", help="CSV with budget items")
    p.add_argument("--departments", default="data/sample/departments.csv", help="CSV with department limits")
    p.add_argument("--budget", type=float, default=1000.0, help="total budget, million KZT")
    p.add_argument(
        "--method",
        choices=["expected", "robust", "recession", "stable", "growth"],
        default="expected",
        help="optimisation criterion",
    )
    p.add_argument("--out", type=Path, help="optional path to save the allocation as CSV")
    args = p.parse_args(argv)

    problem = load_problem(args.items, args.departments, args.budget)
    if args.method == "expected":
        sol = solve_expected(problem)
    elif args.method == "robust":
        sol = solve_robust(problem)
    else:
        sol = solve_scenario(problem, args.method)

    table = sol.to_frame(problem)
    print(table.to_string(index=False))
    print()
    print(sol.by_department(problem).round(2).to_string())
    print()
    print(f"method            : {sol.method}")
    print(f"objective         : {sol.objective:,.2f}")
    print(f"allocated / budget: {sol.total_allocated:,.2f} / {args.budget:,.2f}")
    print(f"budget shadow price: {sol.budget_shadow_price:.4f}")
    print("effect by scenario: " + ", ".join(f"{k}={v:,.1f}" for k, v in sol.effects.items()))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.out, index=False)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
