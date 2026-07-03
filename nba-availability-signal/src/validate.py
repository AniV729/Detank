"""
src/validate.py
===============
Expanding-window walk-forward validation. For each fold we train only on games
that occurred strictly *before* the test window, so there is never any temporal
leakage. Every model is scored out-of-sample and compared against honest
baselines.

Models
------
- base_rate        : predict the training-set event rate for everyone (reference)
- recent_dnp_rate  : use the player's trailing-10 DNP rate as the probability
                     (a free, no-training heuristic a DFS player could eyeball)
- logreg           : L2 logistic regression (linear baseline)
- gbm              : HistGradientBoosting (the model)

Metrics (all out-of-sample)
---------------------------
- ROC AUC
- PR AUC (average precision) — the honest metric for an ~11% base rate
- Brier score — calibration / probability quality
- Lift @ top decile — of the 10% flagged highest-risk, how many x base rate
  actually sat/were cut (the DFS-actionable number)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import FEATURE_COLS

logger = logging.getLogger(__name__)

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def lift_at_decile(y_true: np.ndarray, y_prob: np.ndarray, top_frac: float = 0.10) -> float:
    """Lift of the top-`top_frac` highest-risk predictions over the base rate."""
    n = len(y_true)
    k = max(1, int(round(n * top_frac)))
    order = np.argsort(y_prob)[::-1][:k]
    precision_topk = y_true[order].mean()
    base = y_true.mean()
    return float(precision_topk / base) if base > 0 else float("nan")


def _metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 0.0, 1.0)
    out: dict[str, float] = {}
    # AUC/AP undefined if a fold has a single class; guard it.
    if len(np.unique(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    out["brier"] = float(brier_score_loss(y_true, y_prob))
    out["lift_top10"] = lift_at_decile(y_true, y_prob)
    out["base_rate"] = float(y_true.mean())
    return out


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _fit_predict_gbm(Xtr, ytr, Xte) -> np.ndarray:
    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, max_depth=None,
        l2_regularization=1.0, random_state=RANDOM_STATE,
    )
    model.fit(Xtr, ytr)
    return model.predict_proba(Xte)[:, 1]


def _fit_predict_logreg(Xtr, ytr, Xte) -> np.ndarray:
    pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    pipe.fit(Xtr, ytr)
    return pipe.predict_proba(Xte)[:, 1]


# ---------------------------------------------------------------------------
# Walk-forward driver
# ---------------------------------------------------------------------------

def run_walk_forward(
    df: pd.DataFrame, n_folds: int = 8, start_fold: int = 2,
) -> dict:
    """
    Expanding-window walk-forward. Returns per-fold metrics, pooled
    out-of-sample predictions, and permutation feature importance.
    """
    df = df.sort_values("game_date").reset_index(drop=True)

    # Contiguous, time-ordered folds by row quantile (dates are sorted).
    fold_id = pd.qcut(np.arange(len(df)), q=n_folds, labels=False)
    df = df.assign(_fold=fold_id)

    X_all = df[FEATURE_COLS]
    y_all = df["label"].to_numpy()

    per_fold: list[dict] = []
    pooled_true: list[np.ndarray] = []
    pooled: dict[str, list[np.ndarray]] = {
        "base_rate": [], "recent_dnp_rate": [], "logreg": [], "gbm": []
    }
    pooled_meta: list[pd.DataFrame] = []

    for f in range(start_fold, n_folds):
        test_mask = df["_fold"] == f
        # strictly earlier games only
        test_start_date = df.loc[test_mask, "game_date"].min()
        train_mask = df["game_date"] < test_start_date
        if train_mask.sum() < 500 or test_mask.sum() < 50:
            continue

        Xtr, ytr = X_all[train_mask], y_all[train_mask]
        Xte, yte = X_all[test_mask], y_all[test_mask]

        preds = {
            "base_rate": np.full(len(yte), ytr.mean()),
            "recent_dnp_rate": np.clip(Xte["dnp_rate_10"].to_numpy(), 0, 1),
            "logreg": _fit_predict_logreg(Xtr, ytr, Xte),
            "gbm": _fit_predict_gbm(Xtr, ytr, Xte),
        }

        row = {"fold": f, "train_n": int(train_mask.sum()), "test_n": int(test_mask.sum()),
               "test_start": str(test_start_date.date())}
        for name, p in preds.items():
            m = _metrics(yte, p)
            for k, v in m.items():
                row[f"{name}_{k}"] = v
            pooled[name].append(p)
        per_fold.append(row)
        pooled_true.append(yte)
        pooled_meta.append(df.loc[test_mask, ["season", "game_date"]])

        logger.info(
            "fold %d | train=%d test=%d (>=%s) | gbm ROC=%.3f PR=%.3f lift@10=%.2f",
            f, row["train_n"], row["test_n"], row["test_start"],
            row["gbm_roc_auc"], row["gbm_pr_auc"], row["gbm_lift_top10"],
        )

    pooled_arrays = {name: np.concatenate(v) for name, v in pooled.items() if v}
    y_true = np.concatenate(pooled_true)
    meta = pd.concat(pooled_meta, ignore_index=True)

    # --- pooled out-of-sample metrics per model ---
    pooled_metrics = {name: _metrics(y_true, p) for name, p in pooled_arrays.items()}

    # --- permutation importance (train on all-but-last fold, test on last) ---
    importance = _permutation_importance(df, n_folds)

    return {
        "per_fold": pd.DataFrame(per_fold),
        "pooled_metrics": pooled_metrics,
        "pooled_true": y_true,
        "pooled_pred": pooled_arrays,
        "pooled_meta": meta,
        "importance": importance,
        "n_rows": len(df),
        "overall_base_rate": float(y_all.mean()),
    }


def _permutation_importance(df: pd.DataFrame, n_folds: int) -> pd.Series:
    last = n_folds - 1
    test_mask = df["_fold"] == last
    test_start = df.loc[test_mask, "game_date"].min()
    train_mask = df["game_date"] < test_start

    model = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.05, l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )
    model.fit(df.loc[train_mask, FEATURE_COLS], df.loc[train_mask, "label"])
    result = permutation_importance(
        model, df.loc[test_mask, FEATURE_COLS], df.loc[test_mask, "label"],
        n_repeats=10, random_state=RANDOM_STATE, scoring="average_precision",
    )
    return pd.Series(result.importances_mean, index=FEATURE_COLS).sort_values(ascending=False)


if __name__ == "__main__":
    import argparse

    from src.data import load_player_logs
    from src.features import build_features
    from src.panel import build_panel

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run walk-forward validation")
    parser.add_argument("--seasons", nargs="+",
                        default=["2021-22", "2022-23", "2023-24", "2024-25"])
    args = parser.parse_args()

    raw = load_player_logs(args.seasons)
    feats = build_features(raw, build_panel(raw))
    res = run_walk_forward(feats)

    print("\n=== Pooled out-of-sample metrics ===")
    for name, m in res["pooled_metrics"].items():
        print(f"{name:16s}  ROC={m['roc_auc']:.3f}  PR={m['pr_auc']:.3f}  "
              f"Brier={m['brier']:.3f}  lift@10={m['lift_top10']:.2f}")
    print(f"\nBase rate: {res['overall_base_rate'] * 100:.2f}%")
    print("\n=== Feature importance (permutation, AP) ===")
    print(res["importance"].to_string())
