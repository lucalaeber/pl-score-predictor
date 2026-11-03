# Static Premier League Score Predictor

A static sports-analytics model that fits team attack/defense ratings from
historical Premier League results (Dixon-Coles MLE) and predicts the most
probable exact scoreline, outcome probabilities, and projected final table
for the real 2026-27 season (all 380 fixtures).

## What it does

1. **Data ingestion** — downloads the five seasons 2021-22 to 2025-26
   (`TRAIN_SEASON_CODES` in `model.py`) from football-data.co.uk as CSV (via
   pandas), plus the in-progress 2026-27 season, which also supplies this
   season's real 20-team membership (promotions/relegations).
2. **Expected goals (xG)** — merges in per-match xG from Understat.com
   (`understat.py`) for the same fixtures, and also folds in the current
   season's played matches (which already include actual goals). Attack,
   defense, and home-advantage are fit against xG rather than the actual
   final score, since xG is a much less noisy measure of how a team actually
   played (a stray deflection or a wasted open goal swings the final score
   without reflecting the run of play).
3. **Dixon-Coles model** — fits per-team attack/defense strengths, a home
   advantage term, and the low-score correlation parameter (rho) by maximum
   likelihood, with exponential time decay (ξ = 0.002/day, tuned via
   backtesting — see below) so recent results count more. Rho is still fit
   against actual scorelines, since the effect it corrects for (0-0/1-0/0-1/
   1-1 being more or less common than independent Poisson predicts) is a
   real discrete-score phenomenon that xG wouldn't show.
4. **Ensemble** (`EnsembleModel` in `model.py`) — fits the model twice (once
   on xG, once on actual goals) and averages their score-matrix predictions.
   Backtesting showed the two make different kinds of mistakes and combining
   them beats either alone on log loss/Brier in every season tested — see
   the backtesting section below.
5. **Promoted clubs** — a club with no recent top-flight history starts from
   the average rating of the team that finished 18th in each training season.
   Its own 2026-27 results are then blended in: a second fit puts the club in
   the model with a Gaussian prior pulling its ratings toward that baseline
   (sd `PROMOTED_PRIOR_SD` = 0.3). Early on the baseline dominates; as
   matches accumulate the club's own results take over. This replaced an
   all-or-nothing switch — see `midseason_backtest.py` below.
6. **Fixture list** — the real official 2026-27 schedule
   (`fixtures_2026_27_raw.json`), not a simulated round-robin. Cross-checked
   against actual recorded 2026-27 results for gameweeks 1-2 (exact match).
7. **Match predictions** (`model.py`) — for each fixture, a Poisson scoreline
   grid (with the Dixon-Coles low-score adjustment), the most likely exact
   score, and Win/Draw/Loss probabilities.
8. **Season projection** (`simulate_season.py`) — Monte Carlo simulation of
   the whole season (20,000 runs by default), sampling one scoreline per
   fixture per run from that fixture's own probability grid, aggregated into
   expected points, expected final position, and title/top-4/relegation
   odds per team.
9. **Next gameweek, injury-adjusted** (`next_gameweek.py`, `injuries.py`) —
   live injury/suspension/doubt news from the free Fantasy Premier League API
   scales down each side's expected goals by the share of its attacking output
   (xG + xA) expected to be missing. **Experimental**: attack only (no free
   signal for defensive importance), and the effect size is an assumption (a
   substitute replaces half of a missing player's output) that can't be
   backtested, because no free source keeps historical injury news.
   `injury_effect.py` checks the assumption against this season's matches: the
   data is consistent with it (and, if anything, suggests a larger effect) but
   with ~180 team-matches it can't pin it down. In practice the adjustment moves
   win probabilities by a few points where a key attacker is out and rarely
   changes a predicted scoreline.

## Usage

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python model.py              # -> predictions_2026_2027.csv
python simulate_season.py    # -> projected_table_2026_2027.csv
python track_record.py       # -> track_record_2026_2027.csv, site/track_record.json
python next_gameweek.py      # -> next_gameweek.csv, site/next_gameweek.json
```

Understat data for completed seasons is cached in `data_cache/` (gitignored)
after the first successful fetch, and requests have timeouts and retries, so
an Understat outage only blocks a run that needs a season it hasn't cached yet
(an in-progress season falls back to its stale cache with a warning).
`python next_gameweek.py --goals-only` skips Understat entirely, for
development while it's unreachable; its output comes from a simplified model
and must not be published as the real thing (the site flags such data).

`predictions_2026_2027.csv` columns:

```
Gameweek, Date, HomeTeam, AwayTeam, Expected_Home_Goals, Expected_Away_Goals,
Most_Likely_Score, P_Home, P_Draw, P_Away, Summary
```

`projected_table_2026_2027.csv` columns:

```
Projected_Rank, Team, Expected_Points, Expected_GD, Expected_Position,
P_Title, P_Top4, P_Relegation, Promoted_No_PL_History
```

## Backtesting (`backtest.py`)

Since we can't know how well the model predicts a season that hasn't been
played yet, `backtest.py` validates it against seasons we *do* know the
outcome of: train on the 3 seasons before a target season, predict every
match of that target season using only information available beforehand,
then compare against what actually happened.

```bash
python backtest.py 2526              # train on 5 prior seasons, test against real 2025-26
python backtest.py 2526 5 compare    # xG-fit vs goals-fit individually, same target
python backtest.py 2526 5 ensemble   # the two combined -- what production actually uses
python backtest.py 2526 3 sweep-xi   # scan decay rates, report which scored best
python backtest.py 2526 5 sweep-l2 0.002  # scan attack/defense shrinkage strength
python backtest.py 2526 5 home-adv   # global vs. per-team home advantage
```

`track_record.py` runs the equivalent check against the *current, in-progress*
2026-27 season: fit on only prior seasons, predict every match played so far,
compare to what actually happened. Re-run it every gameweek or two (then
regenerate `site/track_record.json`) to keep the "Track Record" tab current.

Each run reports exact-scoreline accuracy, match-result (W/D/L) accuracy,
log loss, and Brier score against two things: the actual outcomes, and a
"naive" baseline computed only from the training seasons' base rates (e.g.
always guessing the most common result) — a real forecasting model should
beat that baseline, and if it doesn't, that's a sign something needs fixing.
It also compares the simulated final-table projection against the season's
real final table (rank correlation, points error, top-4/relegation hit rate).

**What backtesting across 2023-24, 2024-25, and 2025-26 found:**

- The originally-used decay rate (ξ = 0.0065/day, a commonly cited value in
  Dixon-Coles writeups) actually *hurt* accuracy here — with a 3-season
  training window, it down-weights the oldest season to near-zero and
  produced an unrealistically low home-advantage estimate and a rho stuck
  near zero. ξ = 0.002 scored better on log loss in all three backtested
  seasons (about 2-3% lower on average) and gives home-advantage/rho values
  in the range the Dixon-Coles literature typically reports.
- Fitting on xG vs. actual goals individually is close to a wash: xG edged
  out actual-goals on exact-score and result accuracy in 2 of 3 seasons,
  while actual-goals was marginally better calibrated (log loss/Brier) in
  all 3 — each makes different mistakes. Averaging the two models' score
  matrices (`EnsembleModel`, `backtest.py`'s `ensemble` mode) beat *both*
  individual models on log loss/Brier in every backtested season (average
  log loss 1.003 → 0.999 versus the xG-only model), so production now fits
  and combines both rather than picking one.
- L2 shrinkage of attack/defense ratings toward the league average (a
  standard fix for overfitting) made essentially no difference (4th-decimal
  changes in log loss) — the model wasn't actually overfitting in the way
  that would predict, so this was dropped rather than added as a "just in
  case" knob.
- Training-window length matters more than expected: average log loss kept
  improving monotonically from 2 through 5 training seasons (1.026 → 1.008 →
  1.005 → 1.003), even with the exponential decay already in place. Bumped
  `TRAIN_SEASON_CODES` from 3 to 5 seasons (2021-22 through 2025-26) as a
  result — worth re-checking whether it keeps improving with 6+ once more
  historical Understat data is worth the extra fetch time.
- Reality check: even after tuning, match-result accuracy sits around
  45-57% depending on the season (a "guess the most common outcome" naive
  baseline gets ~44-45%). That's not a bug — single-match football outcomes
  are inherently noisy, and professional bookmakers with far more data
  (injuries, lineups, market signals) don't dramatically beat this either.
  The season-long projection (`simulate_season.py`) is probably more
  trustworthy than any individual match call, since per-match luck averages
  out over 38 games.
- Re-run `sweep-xi`, `sweep-l2`, `compare`, and `ensemble` against a season
  once it's actually completed (e.g. re-validate against 2026-27 next
  summer) rather than assuming these findings hold indefinitely — the right
  hyperparameters are a property of the data window, not universal
  constants.

### Mid-season replay (`midseason_backtest.py`)

The pre-season backtest can't answer "how should this season's own results
be used as it goes on?", so `midseason_backtest.py` replays five past seasons
(2021-22 to 2025-26, each with a newly promoted club the model had never
seen) at the point the live model is at in November: train on 5 prior seasons
plus the first 90 matches, score the remaining 290. Run on goals-only fits,
since it was built while Understat was unreachable; average log loss:

| variant | log loss | promoted-club matches |
|---|---|---|
| prior seasons only | 1.0090 | 1.0237 |
| + in-season matches (old production) | 0.9996 | 1.0234 |
| ... weighted ×2 / ×3 / ×5 | 0.9977 / 0.9980 / 1.0007 | 1.0253 / 1.0274 / 1.0314 |
| promoted clubs blended, prior sd 0.15 / **0.3** / 0.6 | 0.9979 / **0.9978** / 0.9992 | 1.0111 / **1.0093** / 1.0227 |

- In-season results are clearly worth using (1.0090 → 0.9996).
- Up-weighting them beyond the time decay isn't: ×2 is noise-level, ×5 is worse.
- Blending a promoted club's own results with the baseline beat the old
  all-or-nothing approach at every prior strength, overall and on that club's
  matches. The gain is small and the promoted-club subset (145 matches) is
  about one standard error, so treat it as "consistent and conceptually
  right", not "proven large". Adopted at sd 0.3 (`PROMOTED_PRIOR_SD`).
- Rerun it with Understat reachable to see whether the xG fits agree.

### Injury effect check (`injury_effect.py`)

Using FPL per-match minutes and xG for the 180 team-matches played so far, it
measures how much a team's xG falls relative to a pre-season model's
expectation per unit of attacking output (share of team xG + xA) missing.
Both a minutes-based and a stricter "regulars who barely played" definition
land above the assumed effect (8.9% and 11.6% xG per 10% missing, versus 8.1%
assumed; 90% intervals roughly 4-13% and 6-17%). One structural point came
out of it: xG + xA counts every chance twice (shooter and creator), so a
player's share of xG + xA overstates his share of the team's chances by about
1.6x; `injuries.py` measures that ratio from the data instead of assuming it.
The substitute-replacement share (0.5) was set after looking at these
results, so "inside the interval" is a consistency check, not independent
validation.

## Interactive view

`site/index.html` has four tabs: Fixtures (all 380 matches by gameweek,
searchable), Next Gameweek (the injury-adjusted view, from
`next_gameweek.py`; the tab only appears when `site/next_gameweek.json`
exists), Projected Table (the season simulation), and Track Record
(pre-season predictions vs. actual results for matches played so far, from
`track_record.py`). It reads `site/predictions.json`, `site/table.json`,
`site/track_record.json` and `site/next_gameweek.json`; regenerate those from
the CSVs after re-running the scripts — see the export snippet in the commit
that added the projected-table view, or ask for it again if needed.

## Notes

- Data sources: football-data.co.uk (results, current-season team list),
  Understat.com (xG, via its same-origin JSON endpoint — see `understat.py`
  for details; no login or paid access involved) and the Fantasy Premier
  League API (live player availability; free, public JSON).
- Per-team home advantage (`DixonColes(teams, team_home_adv=True)`,
  `backtest.py`'s `home-adv` mode) was tried and tested inconclusive: it won
  clearly in one backtested season, lost in another, and was a wash in the
  third — average log loss across all three barely moved (1.0029 → 1.0025).
  Not worth ~19 extra parameters for that, so it's left as an opt-in feature
  rather than the default.
- Player-level squad ratings (rating a squad from its players' xG/xA, to
  capture summer transfers and promoted clubs) were considered and left
  unbuilt on purpose: FPL only has current-season totals plus prior *Premier
  League* seasons (so a promoted club's new signings have no history), and
  there are no free historical squad snapshots to backtest against. It would
  need Understat's per-season player data to be testable across past seasons;
  the idea is worth revisiting once that's reachable.
- Free options not yet tried: a short-term "current form" component (e.g.
  an Elo-style adjustment from the last 5-6 games) blended with the long-run
  rating, which might capture momentum/injuries the smooth exponential
  decay misses; incorporating shots/corners data (already in the
  football-data.co.uk CSVs) as an auxiliary signal. Beyond free options,
  bookmaker odds are the single strongest predictor of match outcomes in
  the literature, but they're only available close to kickoff for future
  fixtures (a free-tier odds API could work for a live, closer-to-matchday
  version of this, unlike the current one-shot pipeline).
- Whatever's tried, validate it with `backtest.py` before trusting it —
  several plausible-sounding ideas here (L2 shrinkage, per-team home
  advantage) tested out as no better than what was already there. That's a
  useful result too, not a wasted experiment: it stops the same idea from
  being re-tried later on a hunch.
