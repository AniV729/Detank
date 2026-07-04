"""
src/panel.py
============
Reconstruct a COMPLETE player x team-game panel from the "played-only" box-score
feed.

The NBA Stats player log only contains rows for players who actually played, so
DNPs are invisible. We recover them: for each (player, team) we take the span
between their first and last appearance for that team (their "tenure"), and mark
every team game inside that span where the player has *no* box-score row as a
did-not-play (minutes = 0, played = 0).

This yields objective, self-labeling availability data — no hand labels.

    from src.panel import build_panel
    panel = build_panel(raw_logs_df)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _parse_minutes(series: pd.Series) -> pd.Series:
    """Coerce the MIN column to float. Handles ints and 'MM:SS' strings."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)

    def _one(val: object) -> float:
        if pd.isna(val):
            return np.nan
        s = str(val)
        if ":" in s:
            mm, ss = s.split(":")[:2]
            try:
                return float(mm) + float(ss) / 60.0
            except ValueError:
                return np.nan
        try:
            return float(s)
        except ValueError:
            return np.nan

    return series.map(_one)


def build_panel(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build the full player x team-game panel with DNPs inferred.

    Returns one row per (season, team, player, game) within the player's tenure,
    with columns:
        season, player_id, player_name, team_id, team_abbrev,
        game_id, game_date, matchup, team_wl, minutes, played
    """
    df = raw.copy()
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df["minutes"] = _parse_minutes(df["MIN"])
    # A row exists => the player played (even 0-minute garbage-time cameos are rare
    # and still "available"); treat presence as played.
    df["minutes"] = df["minutes"].fillna(0.0)

    # ------------------------------------------------------------------
    # 1. Team schedule: every (season, team, game) with shared team context.
    #    WL / MATCHUP are identical across all players on a team in a game,
    #    so we can recover them even for the DNP rows we synthesize.
    # ------------------------------------------------------------------
    team_sched = (
        df.sort_values("GAME_DATE")
        .groupby(["SEASON", "TEAM_ID", "GAME_ID"], as_index=False)
        .agg(
            team_abbrev=("TEAM_ABBREVIATION", "first"),
            game_date=("GAME_DATE", "first"),
            matchup=("MATCHUP", "first"),
            team_wl=("WL", "first"),
        )
    )
    logger.info("Team schedule: %d team-games", len(team_sched))

    # ------------------------------------------------------------------
    # 2. Player tenure per (season, team): first & last appearance dates.
    # ------------------------------------------------------------------
    tenure = (
        df.groupby(["SEASON", "TEAM_ID", "PLAYER_ID"], as_index=False)
        .agg(
            player_name=("PLAYER_NAME", "first"),
            first_date=("GAME_DATE", "min"),
            last_date=("GAME_DATE", "max"),
        )
    )
    logger.info("Player-team tenures: %d", len(tenure))

    # ------------------------------------------------------------------
    # 3. Expand: cross each tenure with its team's schedule, keep games
    #    inside the tenure window. This is the set of games the player
    #    *could* have played.
    # ------------------------------------------------------------------
    expanded = tenure.merge(team_sched, on=["SEASON", "TEAM_ID"], how="left")
    in_window = (
        (expanded["game_date"] >= expanded["first_date"])
        & (expanded["game_date"] <= expanded["last_date"])
    )
    expanded = expanded[in_window].copy()
    logger.info("Expanded panel (within tenure): %d rows", len(expanded))

    # ------------------------------------------------------------------
    # 4. Join actual minutes; absence within tenure => DNP.
    # ------------------------------------------------------------------
    played = df[["SEASON", "TEAM_ID", "PLAYER_ID", "GAME_ID", "minutes"]].copy()
    panel = expanded.merge(
        played, on=["SEASON", "TEAM_ID", "PLAYER_ID", "GAME_ID"], how="left"
    )
    panel["played"] = panel["minutes"].notna().astype(int)
    panel["minutes"] = panel["minutes"].fillna(0.0)

    panel = panel.rename(columns={
        "SEASON": "season",
        "PLAYER_ID": "player_id",
        "TEAM_ID": "team_id",
        "GAME_ID": "game_id",
    })

    out_cols = [
        "season", "player_id", "player_name", "team_id", "team_abbrev",
        "game_id", "game_date", "matchup", "team_wl", "minutes", "played",
    ]
    panel = panel[out_cols].sort_values(
        ["season", "player_id", "game_date"]
    ).reset_index(drop=True)

    dnp_rate = 1.0 - panel["played"].mean()
    logger.info("Final panel: %d rows | inferred DNP rate: %.1f%%",
                len(panel), dnp_rate * 100)
    return panel


if __name__ == "__main__":
    import argparse

    from src.data import DEFAULT_SEASONS, load_player_logs

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Build player-game panel")
    parser.add_argument("--seasons", nargs="+", default=DEFAULT_SEASONS)
    args = parser.parse_args()

    raw = load_player_logs(args.seasons)
    panel = build_panel(raw)
    print("\n--- Panel sample ---")
    print(panel.head(8).to_string())
    print(f"\nRows: {len(panel):,}")
    print(f"Played: {panel['played'].mean() * 100:.1f}%  |  "
          f"DNP: {(1 - panel['played'].mean()) * 100:.1f}%")
    print(f"Unique players: {panel['player_id'].nunique():,}")
