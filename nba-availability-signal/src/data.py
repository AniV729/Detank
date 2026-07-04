"""
src/data.py
===========
Ingest full-season NBA player game logs from the NBA Stats API and cache them
to Parquet. One API call per season returns every player-game row for the
regular season, so this is fast and rate-limit friendly.

Usage
-----
    python -m src.data --seasons 2021-22 2022-23 2023-24 2024-25

Programmatic
------------
    from src.data import load_player_logs
    df = load_player_logs(["2023-24"])   # pulls + caches if missing
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"

# The seasons the published results are built on (2010-11 through 2024-25).
DEFAULT_SEASONS = [
    "2010-11", "2011-12", "2012-13", "2013-14", "2014-15", "2015-16",
    "2016-17", "2017-18", "2018-19", "2019-20", "2020-21", "2021-22",
    "2022-23", "2023-24", "2024-25",
]

# Columns we keep from the raw LeagueGameLog player feed.
_KEEP_COLS = [
    "SEASON_ID", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION",
    "GAME_ID", "GAME_DATE", "MATCHUP", "WL", "MIN", "PTS", "PLUS_MINUS",
]


def _cache_path(season: str) -> Path:
    return DATA_DIR / f"player_logs_{season}.parquet"


def fetch_season(season: str, max_retries: int = 4, base_delay: float = 2.0) -> pd.DataFrame:
    """
    Fetch one season of player game logs with exponential-backoff retries.

    Returns a normalized DataFrame (one row per player per game they played).
    """
    from nba_api.stats.endpoints import LeagueGameLog

    last_exc: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            logger.info("Fetching player logs for %s (attempt %d)", season, attempt)
            raw = LeagueGameLog(
                season=season,
                player_or_team_abbreviation="P",
                season_type_all_star="Regular Season",
                timeout=60,
            ).get_data_frames()[0]
            df = raw[[c for c in _KEEP_COLS if c in raw.columns]].copy()
            df["SEASON"] = season
            df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
            logger.info("  -> %d player-game rows for %s", len(df), season)
            return df
        except Exception as exc:  # noqa: BLE001 - network flakiness, retry
            last_exc = exc
            delay = base_delay * (2 ** (attempt - 1))
            logger.warning("  fetch failed (%s); retrying in %.1fs", exc, delay)
            time.sleep(delay)

    raise RuntimeError(f"Failed to fetch season {season} after {max_retries} attempts") from last_exc


def load_player_logs(seasons: list[str], refresh: bool = False) -> pd.DataFrame:
    """
    Load player logs for the given seasons, pulling + caching any that are missing.

    Parameters
    ----------
    seasons : list[str]
        Season strings, e.g. ["2022-23", "2023-24"].
    refresh : bool
        If True, re-fetch even when a cached Parquet exists.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []

    for season in seasons:
        path = _cache_path(season)
        if path.exists() and not refresh:
            logger.info("Loading cached %s", path.name)
            frames.append(pd.read_parquet(path))
            continue

        df = fetch_season(season)
        df.to_parquet(path, index=False)
        logger.info("Cached %s", path.name)
        frames.append(df)
        time.sleep(1.0)  # be polite between season calls

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d total player-game rows across %d seasons",
                len(combined), len(seasons))
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache NBA player game logs")
    parser.add_argument(
        "--seasons", nargs="+", default=DEFAULT_SEASONS,
        help="Seasons to pull, e.g. 2022-23 2023-24",
    )
    parser.add_argument("--refresh", action="store_true", help="Re-fetch even if cached")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    df = load_player_logs(args.seasons, refresh=args.refresh)
    print(f"\nCached {len(df):,} player-game rows across seasons: {', '.join(args.seasons)}")
    print(f"Date range: {df['GAME_DATE'].min().date()} -> {df['GAME_DATE'].max().date()}")


if __name__ == "__main__":
    main()
