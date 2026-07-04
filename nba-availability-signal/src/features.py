"""
src/features.py
===============
Turn the reconstructed panel into a modeling table of STRICTLY CAUSAL features
(only information available before tip-off) plus an OBJECTIVE label.

Label (the thing we predict)
----------------------------
For a rotation player heading into a game:

    event = 1  if the player DNPs, OR plays < 50% of their trailing-10 minutes
    event = 0  otherwise

The trailing baseline is computed from games strictly *before* the current one,
so the label never peeks at the present. We only score **rotation players**
(trailing-10 average >= 20 min) with at least 5 games of history, because a
"minutes drop" is only meaningful for someone with an established role.

Every feature is `.shift(1)`-ed within its group so the current game is excluded.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

ROTATION_MIN = 20.0     # trailing-10 avg minutes to count as a rotation player
EVENT_FRACTION = 0.5    # minutes below this fraction of baseline = drop event
MIN_HISTORY = 5         # require >= this many prior games for a valid baseline

FEATURE_COLS = [
    # team context
    "team_rest_days", "back_to_back", "team_win_pct_10",
    "team_avg_margin_5", "team_blowout_losses_5", "team_game_num",
    # player context
    "trailing_min_5", "trailing_min_10", "min_std_10", "dnp_rate_10",
    "prev_min", "prev_played", "games_last_14d",
]


# ---------------------------------------------------------------------------
# Team-game margins (recovered from the shared GAME_ID across both teams)
# ---------------------------------------------------------------------------

def _team_game_margins(raw: pd.DataFrame) -> pd.DataFrame:
    """Compute per team-game points, opponent points, and margin from the feed."""
    df = raw.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    team_pts = (
        df.groupby(["SEASON", "TEAM_ID", "GAME_ID"], as_index=False)
        .agg(team_pts=("PTS", "sum"), game_date=("GAME_DATE", "first"))
    )
    game_total = (
        df.groupby(["SEASON", "GAME_ID"], as_index=False)
        .agg(game_total=("PTS", "sum"))
    )
    tg = team_pts.merge(game_total, on=["SEASON", "GAME_ID"], how="left")
    tg["opp_pts"] = tg["game_total"] - tg["team_pts"]
    tg["margin"] = tg["team_pts"] - tg["opp_pts"]
    return tg[["SEASON", "TEAM_ID", "GAME_ID", "game_date", "margin"]]


def _team_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Rolling team-level features, all shifted to exclude the current game."""
    tg = _team_game_margins(raw).sort_values(["SEASON", "TEAM_ID", "game_date"])
    g = tg.groupby(["SEASON", "TEAM_ID"], group_keys=False)

    tg["team_rest_days"] = g["game_date"].transform(
        lambda s: s.diff().dt.days
    ).fillna(3).clip(upper=10)
    tg["back_to_back"] = (tg["team_rest_days"] == 1).astype(int)

    tg["win_flag"] = (tg["margin"] > 0).astype(int)
    tg["team_win_pct_10"] = g["win_flag"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=3).mean()
    )
    tg["team_avg_margin_5"] = g["margin"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=2).mean()
    )
    tg["blowout_loss"] = (tg["margin"] <= -20).astype(int)
    tg["team_blowout_losses_5"] = g["blowout_loss"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).sum()
    )
    tg["team_game_num"] = g.cumcount() + 1

    return tg.rename(columns={"SEASON": "season", "TEAM_ID": "team_id", "GAME_ID": "game_id"})[
        ["season", "team_id", "game_id", "team_rest_days", "back_to_back",
         "team_win_pct_10", "team_avg_margin_5", "team_blowout_losses_5", "team_game_num"]
    ]


# ---------------------------------------------------------------------------
# Player-level features + label
# ---------------------------------------------------------------------------

def build_features(raw: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    """
    Produce the modeling table: identifiers + FEATURE_COLS + 'label' + 'game_date'.
    Only rotation-player rows with sufficient history are returned.
    """
    p = panel.sort_values(["season", "player_id", "game_date"]).copy()
    grp = p.groupby(["season", "player_id"], group_keys=False)

    # --- Player rolling features (shifted: exclude current game) ---
    p["trailing_min_5"] = grp["minutes"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=MIN_HISTORY).mean()
    )
    p["trailing_min_10"] = grp["minutes"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=MIN_HISTORY).mean()
    )
    p["min_std_10"] = grp["minutes"].transform(
        lambda s: s.shift(1).rolling(10, min_periods=MIN_HISTORY).std()
    )
    p["dnp_rate_10"] = grp["played"].transform(
        lambda s: (1 - s).shift(1).rolling(10, min_periods=MIN_HISTORY).mean()
    )
    p["prev_min"] = grp["minutes"].transform(lambda s: s.shift(1))
    p["prev_played"] = grp["played"].transform(lambda s: s.shift(1))
    p["prior_games"] = grp.cumcount()

    # games actually played in the trailing 14 days (load/fatigue), excl. current
    p = p.sort_values(["season", "player_id", "game_date"])
    rolled = (
        p.set_index("game_date")
        .groupby(["season", "player_id"])["played"]
        .rolling("14D").sum()
        .reset_index(level=[0, 1], drop=True)
        .to_numpy()
    )
    p["games_last_14d"] = rolled - p["played"].to_numpy()

    # --- Merge team features ---
    tf = _team_features(raw)
    p = p.merge(tf, on=["season", "team_id", "game_id"], how="left")

    # --- Objective label (uses only the prior-games baseline) ---
    baseline = p["trailing_min_10"]
    drop_event = p["minutes"] < (EVENT_FRACTION * baseline)
    dnp_event = p["played"] == 0
    p["label"] = (drop_event | dnp_event).astype(int)

    # --- Restrict to rotation players with enough history ---
    is_rotation = baseline >= ROTATION_MIN
    has_history = p["prior_games"] >= MIN_HISTORY
    model_df = p[is_rotation & has_history].copy()

    # any remaining NaNs in features -> fill neutral (HistGBM also handles NaN,
    # but keep it explicit for baseline models)
    for col in FEATURE_COLS:
        if col not in model_df.columns:
            raise KeyError(f"expected feature '{col}' missing")

    keep = ["season", "player_id", "player_name", "team_abbrev",
            "game_id", "game_date"] + FEATURE_COLS + ["label", "minutes"]
    model_df = model_df[keep].reset_index(drop=True)

    logger.info(
        "Modeling table: %d rows | rotation players | event rate %.1f%%",
        len(model_df), model_df["label"].mean() * 100,
    )
    return model_df


if __name__ == "__main__":
    import argparse

    from src.data import DEFAULT_SEASONS, load_player_logs
    from src.panel import build_panel

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Build modeling features")
    parser.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    args = parser.parse_args()

    raw = load_player_logs(args.seasons)
    panel = build_panel(raw)
    feats = build_features(raw, panel)

    print("\n--- Feature table sample ---")
    print(feats.head(6).to_string())
    print(f"\nRows: {len(feats):,}  |  event rate: {feats['label'].mean() * 100:.2f}%")
    print(f"Date range: {feats['game_date'].min().date()} -> {feats['game_date'].max().date()}")
    print("\nNull counts per feature:")
    print(feats[FEATURE_COLS].isna().sum().to_string())
