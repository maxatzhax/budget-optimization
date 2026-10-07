"""Generate the synthetic sample dataset in data/sample/.

Usage:
    python scripts/generate_synthetic_data.py --seed 42
"""

import argparse
from pathlib import Path

from budget_opt.data import generate_synthetic

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=ROOT / "data" / "sample")
    args = p.parse_args()

    items, departments = generate_synthetic(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    items.to_csv(args.out_dir / "budget_items.csv", index=False)
    departments.to_csv(args.out_dir / "departments.csv", index=False)
    print(f"{len(items)} items, {len(departments)} departments -> {args.out_dir}")


if __name__ == "__main__":
    main()
