# NBA Player Availability Signal

> Predicting, *before tip-off*, which NBA rotation players will be **rested or have their minutes cut** — trained and validated on 15 seasons of real data.

---

## The problem in plain English

NBA teams don't always try to win. Stars sit on back-to-backs to manage their bodies ("load management"), and veterans on bad teams get quietly shut down late in the season so the team can lose and improve its draft pick ("tanking"). When a player unexpectedly sits or plays 12 minutes instead of 34, it matters — especially to **daily-fantasy players**, who lose their entry if a star they picked doesn't play.

The official NBA injury report is **reactive**: it often confirms a rest decision only a few hours before the game. This project asks a harder, more useful question:

> **Can we predict, using only information known *before* the game, that a normally-heavy-minutes player is about to sit or get cut down?**

The answer is yes — meaningfully better than chance, and consistently so across 15 seasons.

---

## How it works (the 60-second version)

The whole system is a five-step assembly line. Here's the intuition for each step; the detailed mechanics are in the code under `src/`.

**1. Get the data.** We download every player's box score for every regular-season game, 2010–2025 — who played, how many minutes, how many points.

**2. Find the "did not play" games.** A box score only lists players who *played*. So if a regular is missing from a game, that's an invisible DNP. We reconstruct those: for each player we look at the window between their first and last game on a team, and any team game in that window where they have no box-score row becomes a "rested / DNP" record. This turns the raw feed into a **complete** record of played-vs-sat.

**3. Describe each situation using only the past.** For every player heading into every game, we build a set of clues that were knowable *before* tip-off — for example:
   - How many minutes they've been averaging lately (their "role")
   - How *erratic* their minutes have been
   - How often they've sat recently
   - Whether it's a back-to-back / how rested the team is
   - How the team's been playing (win rate, blowouts)
   - How far into the season it is (rest ramps up late)

   Crucially, **none of these clues use the current game's outcome** — otherwise we'd be cheating.

**4. Define what we're predicting.** The thing we want to flag ("the event") is objective: a player **sits, OR plays less than half their recent average minutes**. We only score *rotation players* (recent average ≥ 20 min), because a "minutes drop" only means something for someone with an established role. About **9%** of these player-games are events.

**5. Learn the pattern and test it honestly.** A machine-learning model learns which combinations of clues tend to precede an event. Then — and this is the important part — we test it **walk-forward**: train only on the past, predict an unseen future stretch, then roll forward. The model is *never* graded on data it trained on. We repeat this across 10 chronological chunks from 2013 to 2025.

The output for each player-game is a **probability** (0–100%) that they're about to sit or get cut.

---

## Does it actually work?

Yes — and the numbers are honest (realistic, not suspiciously perfect), stable across 15 seasons, and clearly beat simple baselines.

| Model | ROC-AUC | PR-AUC | Brier | Lift @ top decile |
|-------|:------:|:------:|:-----:|:-----------------:|
| Guess the base rate | 0.55 | 0.10 | 0.083 | 1.0× |
| "How often they sat lately" heuristic | 0.65 | 0.16 | 0.083 | 2.5× |
| Logistic regression (simple model) | 0.74 | 0.36 | 0.176 | 4.5× |
| **Gradient Boosting (the model)** | **0.75** | **0.39** | **0.068** | **4.5×** |

**The headline, in plain words:** if you take the **10% of players the model is most worried about** on a given night, about **40% of them actually sit or get their minutes cut** — versus a **9% base rate** if you picked at random. That's a **4.5× lift**, and it's the number a daily-fantasy user would actually act on.

The model's probabilities are also **well-calibrated** — when it says "30% chance," it happens about 30% of the time — which is why it beats the logistic baseline despite similar ranking power.

### Figures

| | |
|---|---|
| ![Precision-Recall](results/pr_curves.png) | ![Calibration](results/calibration.png) |
| ![Per-fold stability](results/per_fold.png) | ![Cumulative gains](results/lift_curve.png) |

![Feature importance](results/importance.png)

The most predictive clue by far is **the player's minutes in their most recent game**, followed by their recent minutes level, how far into the season it is, and how erratic their minutes have been.

---

## Data scope

Fifteen NBA regular seasons, **2010-11 → 2024-25** (Oct 2010 → Apr 2025). Regular season only — playoffs and preseason are excluded because rotations behave differently there and would pollute the signal.

The unit of analysis is a **player-game observation** (one row per player per game), not a game:

| Level | Count |
|-------|------:|
| Unique games (~1,230/season × 15) | ~17,900 |
| Team-games (each game ×2) | 35,776 |
| Player-games actually played | 377,780 |
| Player-games incl. inferred DNPs | 503,683 |
| **Modeling rows** (rotation players, ≥5 games history) | **202,796** |

---

## Honest limitations

- The box score only lists players who **played**, so DNPs are *inferred* from absence. This lumps together rest, injury, and coach's decisions — the model predicts **unavailability of any kind**, not tanking specifically. Separating rest from injury needs official injury-report data (on the roadmap).
- Mid-season trades and G-League movement can create false "absences"; restricting each player to their first–last-game window reduces this but doesn't fully remove it.
- A meaningful share of minutes drops (sudden injuries, in-game ejections) is genuinely unpredictable from schedule/role/form alone — which is exactly why ROC ~0.75, not ~0.95, is the *believable* result.

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

Outputs land in `results/` (metrics JSON + PNG plots). The whole thing runs in about a minute on cached data.

### Project layout

```
src/data.py       # download + cache season box scores
src/panel.py      # reconstruct played-vs-sat for every player-game (recovers DNPs)
src/features.py   # build pre-game features + the objective label
src/validate.py   # walk-forward training/testing + baselines + metrics
src/plots.py      # figures
run.py            # runs the whole pipeline end to end
```

## Roadmap

- [ ] Ingest historical **official injury reports** to (a) separate rest from injury and (b) measure how much *earlier* this model flags shutdowns than the injury report does.
- [ ] Per-player minutes **regression** (predict the actual number, not just the drop flag).
- [ ] Bootstrap **confidence intervals** + a permutation **null test** on the metrics.
- [ ] Daily-slate dashboard surfacing top-N risk players and the backups who'd absorb their minutes.
