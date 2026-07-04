"""
run.py
======
End-to-end pipeline: load -> panel -> features -> walk-forward -> metrics + plots.

    python run.py
    python run.py --seasons 2022-23 2023-24 2024-25

Writes:
    results/metrics.json      pooled + per-fold metrics
    results/per_fold.csv      full per-fold table
    results/*.png             figures
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from src.data import DEFAULT_SEASONS, load_player_logs
from src.features import build_features
from src.panel import build_panel
from src.plots import generate_all
from src.validate import run_walk_forward

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run the full availability-signal pipeline")
    parser.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    parser.add_argument("--folds", type=int, default=12)
    args = parser.parse_args()

    raw = load_player_logs(args.seasons)
    panel = build_panel(raw)
    feats = build_features(raw, panel)
    res = run_walk_forward(feats, n_folds=args.folds)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # --- metrics.json ---
    payload = {
        "seasons": args.seasons,
        "n_model_rows": int(res["n_rows"]),
        "overall_base_rate": res["overall_base_rate"],
        "pooled_metrics": res["pooled_metrics"],
        "importance": res["importance"].round(4).to_dict(),
        "n_folds_evaluated": int(len(res["per_fold"])),
    }
    (RESULTS_DIR / "metrics.json").write_text(json.dumps(payload, indent=2))

    # --- per_fold.csv ---
    res["per_fold"].to_csv(RESULTS_DIR / "per_fold.csv", index=False)

    # --- plots ---
    generate_all(res, RESULTS_DIR)

    # --- console summary ---
    print("\n" + "=" * 60)
    print("POOLED OUT-OF-SAMPLE RESULTS (walk-forward)")
    print("=" * 60)
    print(f"{'model':18s}{'ROC':>8s}{'PR-AUC':>9s}{'Brier':>8s}{'lift@10':>9s}")
    for name, m in res["pooled_metrics"].items():
        print(f"{name:18s}{m['roc_auc']:>8.3f}{m['pr_auc']:>9.3f}"
              f"{m['brier']:>8.3f}{m['lift_top10']:>9.2f}")
    print(f"\nBase rate: {res['overall_base_rate'] * 100:.2f}%  |  "
          f"rows: {res['n_rows']:,}  |  folds: {len(res['per_fold'])}")
    print(f"Results written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
