# NBA Player Availability Signal

> Predicting, *before tip-off*, which rotation players will be **rested or have their minutes cut** — validated with walk-forward testing on real ground truth.

---

## Why this exists

NBA rosters are increasingly managed for *load* and *draft position*, not just for winning tonight. Star players sit on back-to-backs; veterans on eliminated or tanking teams get quietly shut down. These "healthy DNPs" are worth real money to daily-fantasy players and are systematically **under-predicted by official injury reports**, which are reactive and often published only hours before tip-off.

This project builds a **forward-looking availability model**: for every rotation player heading into their team's next game, it estimates the probability of a *significant minutes drop or DNP*, using only information known before the game.

Unlike a "detect tanking after the fact" heuristic, the target here is **objective and self-labeling** — we simply observe how many minutes the player actually ended up playing. No hand-picked labels, no circular reasoning.

---

## The signal in one line

For each `(player, game)` with only pre-game information:

```
P( next-game minutes < 0.5 x trailing-10-game average, OR did-not-play )
```

Everything is validated with **expanding-window walk-forward** splits (train on the past, predict the unseen future) and reported against a naive baseline — never in-sample.

---

## Method

| Stage | What happens |
|-------|--------------|
| **Ingest** | Full-season player game logs from the NBA Stats API (one call per season, cached to Parquet) |
| **Panel reconstruction** | Rebuild a *complete* player x team-game panel, inferring DNPs (rotation players absent from the box score within their tenure) |
| **Features** | Strictly causal, pre-game features: rest days, back-to-backs, trailing minutes/volatility, recent role, team form, blowout exposure, season phase |
| **Labels** | Objective: actual next-game minutes vs. the player's own trailing baseline |
| **Validate** | Expanding-window walk-forward; calibrated gradient-boosted classifier; per-fold + pooled metrics vs. baseline |

## Honest limitations (documented up front)

- The box score only lists players who **played**, so DNPs are *inferred* from absence within a player's tenure. This conflates rest, injury, and coach's-decision DNPs — the model predicts **unavailability of any kind**, not tanking specifically. This is stated deliberately: separating rest from injury requires official injury-report ingestion (roadmap below).
- Mid-season trades and G-League two-way movement can create false "absences"; tenure-clipping mitigates but does not eliminate this.
- This is a **medium-frequency roster signal**, not a tick-level trading edge.

---

## Data scope

Fifteen NBA regular seasons, **2010-11 → 2024-25** (game dates Oct 2010 → Apr 2025). Regular season only — playoffs/preseason are excluded because their rotation logic is different and would contaminate the label.

The unit of analysis is a **player-game observation** (one row per player per game), *not* a game. The counts cascade like this:

| Level | Count | Meaning |
|-------|------:|---------|
| Unique games | ~17,900 | 1,230 games/season × 15 (adjusted for lockout/COVID-shortened years) |
| Team-games | 35,776 | each game counted once per team |
| Player-games (played) | 377,780 | box-score rows where the player appeared |
| Player-games (+ inferred DNPs) | 503,683 | after adding rest/DNP rows within each player's tenure |
| **Modeling rows** | **202,796** | rotation players (trailing-10 ≥ 20 min) with ≥ 5 games of history |

## Results

Validated on **202,796 rotation-player-game observations**, with an **8.9% event base rate**. All numbers below are **out-of-sample**, pooled across 10 expanding-window walk-forward folds spanning **2013 → 2025** — the model never sees the period it's scored on.

| Model | ROC-AUC | PR-AUC | Brier | Lift @ top decile |
|-------|:------:|:------:|:-----:|:-----------------:|
| Base rate (reference) | 0.547 | 0.104 | 0.083 | 1.25x |
| Recent-DNP heuristic | 0.653 | 0.158 | 0.083 | 2.53x |
| Logistic regression | 0.744 | 0.362 | **0.176** | 4.45x |
| **Gradient Boosting** | **0.752** | **0.390** | **0.068** | **4.49x** |

**What the headline number means:** of the 10% of players the model flags as highest-risk on a given slate, **~40% actually sit or get their minutes cut** — a **4.5x lift** over the 8.9% base rate. The gradient-booster also cuts the Brier score by ~60% vs. the logistic baseline, so its probabilities are usable as-is (no post-hoc calibration needed).

**Stability:** performance is consistent across every fold from 2013 to 2025 (ROC **0.739–0.763**), spanning the pre- and post-"load-management" eras — not driven by one lucky period or regime.

**Honesty note:** ROC ~0.75 on an inferred, noisy label is a *realistic* number, not a suspiciously clean one. A meaningful share of a player's minutes drop is genuinely unpredictable from schedule/role/form alone — which is exactly why official injury reports (roadmap) would be the next lift.

### Figures

| | |
|---|---|
| ![Precision-Recall](results/pr_curves.png) | ![Calibration](results/calibration.png) |
| ![Per-fold stability](results/per_fold.png) | ![Cumulative gains](results/lift_curve.png) |

![Feature importance](results/importance.png)

Feature importance (permutation, out-of-sample) is led by the player's **most-recent-game minutes** and recent minutes level, then **season progression** (`team_game_num` — rest ramps up late-season, consistent with load management and tanking) and minutes **volatility** (inconsistent-usage players are the most predictable rest candidates), then team rest days / back-to-backs.

### Resume bullet (real numbers)

> Built an end-to-end NBA player-availability model on 200K+ rotation-player-game observations spanning 15 seasons (2010-11–2024-25), engineering strictly causal features from reconstructed box-score panels (recovering inferred DNPs) and validating with expanding-window walk-forward testing; the gradient-boosted classifier reached **0.75 ROC-AUC / 0.39 PR-AUC** out-of-sample with a **4.5x top-decile lift** over base rate and a well-calibrated 0.068 Brier score, beating naive and logistic baselines consistently across all 10 folds (ROC 0.74–0.76, 2013–2025).

---

## Reproduce it

```bash
pip install -r requirements.txt

# 1. Pull + cache raw season logs (defaults to all 15 seasons, one API call each)
python -m src.data

# 2. Run the full pipeline: panel -> features -> walk-forward -> metrics + plots
python run.py

# (both default to 2010-11 .. 2024-25; pass --seasons to override)
```

Outputs land in `results/` (metrics JSON + PNG plots).

---

## Roadmap

- [ ] Ingest historical **official injury reports** to (a) separate rest from injury and (b) benchmark how much *earlier* this model flags shutdowns.
- [ ] Per-player minutes **regression** (not just the drop classifier) for direct DFS projections.
- [ ] Daily-slate dashboard + "value play" surfacing (backup minutes when a starter sits).
