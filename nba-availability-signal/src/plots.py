"""
src/plots.py
============
Generate the figures that go in the README / results folder from a walk-forward
results dict (see src.validate.run_walk_forward).
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import precision_recall_curve  # noqa: E402

logger = logging.getLogger(__name__)

_LABELS = {
    "gbm": "Gradient Boosting",
    "logreg": "Logistic (baseline)",
    "recent_dnp_rate": "Recent-DNP heuristic",
    "base_rate": "Base rate",
}
_COLORS = {
    "gbm": "#1f77b4",
    "logreg": "#ff7f0e",
    "recent_dnp_rate": "#2ca02c",
    "base_rate": "#999999",
}


def plot_pr_curves(res: dict, out: Path) -> None:
    y = res["pooled_true"]
    base = res["overall_base_rate"]
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name in ["gbm", "logreg", "recent_dnp_rate"]:
        p = res["pooled_pred"].get(name)
        if p is None:
            continue
        prec, rec, _ = precision_recall_curve(y, p)
        ap = res["pooled_metrics"][name]["pr_auc"]
        ax.plot(rec, prec, color=_COLORS[name], lw=2,
                label=f"{_LABELS[name]} (AP={ap:.3f})")
    ax.axhline(base, ls="--", color=_COLORS["base_rate"],
               label=f"Base rate ({base * 100:.1f}%)")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Out-of-sample Precision–Recall (walk-forward)")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_calibration(res: dict, out: Path) -> None:
    y = res["pooled_true"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], ls="--", color="#999999", label="Perfect")
    for name in ["gbm", "logreg"]:
        p = res["pooled_pred"].get(name)
        if p is None:
            continue
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        brier = res["pooled_metrics"][name]["brier"]
        ax.plot(mean_pred, frac_pos, "o-", color=_COLORS[name], lw=2,
                label=f"{_LABELS[name]} (Brier={brier:.3f})")
    ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title("Calibration (walk-forward, out-of-sample)")
    ax.legend(loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_per_fold(res: dict, out: Path) -> None:
    pf = res["per_fold"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(pf["test_start"], pf["gbm_roc_auc"], "o-", color=_COLORS["gbm"], label="GBM ROC-AUC")
    ax.plot(pf["test_start"], pf["gbm_pr_auc"], "s-", color="#d62728", label="GBM PR-AUC")
    ax.plot(pf["test_start"], pf["recent_dnp_rate_roc_auc"], "^--",
            color=_COLORS["recent_dnp_rate"], label="Heuristic ROC-AUC")
    ax.set_xlabel("Test fold start date"); ax.set_ylabel("Score")
    ax.set_title("Per-fold out-of-sample performance (stability)")
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 1)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_importance(res: dict, out: Path) -> None:
    imp = res["importance"].sort_values()
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.barh(imp.index, imp.values, color=_COLORS["gbm"])
    ax.set_xlabel("Permutation importance (drop in Average Precision)")
    ax.set_title("Feature importance (GBM, out-of-sample)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def plot_lift(res: dict, out: Path) -> None:
    """Cumulative gains: capture rate vs fraction of players flagged."""
    y = res["pooled_true"]
    p = res["pooled_pred"]["gbm"]
    order = np.argsort(p)[::-1]
    y_sorted = y[order]
    frac_pop = np.arange(1, len(y) + 1) / len(y)
    capture = np.cumsum(y_sorted) / y.sum()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(frac_pop, capture, color=_COLORS["gbm"], lw=2, label="GBM")
    ax.plot([0, 1], [0, 1], ls="--", color="#999999", label="Random")
    ax.axvline(0.10, ls=":", color="#d62728", label="Top decile")
    ax.set_xlabel("Fraction of players flagged (by risk)")
    ax.set_ylabel("Fraction of actual rest/cut events captured")
    ax.set_title("Cumulative gains (walk-forward)")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out, dpi=130); plt.close(fig)


def generate_all(res: dict, out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = {
        "pr_curves.png": plot_pr_curves,
        "calibration.png": plot_calibration,
        "per_fold.png": plot_per_fold,
        "importance.png": plot_importance,
        "lift_curve.png": plot_lift,
    }
    paths: list[Path] = []
    for fname, fn in jobs.items():
        path = out_dir / fname
        fn(res, path)
        logger.info("wrote %s", path.name)
        paths.append(path)
    return paths
