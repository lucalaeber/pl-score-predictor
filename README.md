# Static Premier League Score Predictor

A static sports-analytics model that fits team attack/defense ratings from
historical Premier League results (Dixon-Coles MLE) and predicts the most
probable exact scoreline, outcome probabilities, and projected final table
for the real 2026-27 season (all 380 fixtures).

## What it does

1. **Data ingestion** — downloads 2023-24, 2024-25, and 2025-26 Premier
   League match results directly from football-data.co.uk as CSV (via
   pandas), and separately reads the in-progress 2026-27 season file just to
   determine this season's real 20-team membership (promotions/relegations).
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
4. **Promoted-team baseline** — teams with no recent top-flight history are
   assigned the average rating of the team that finished 18th in each of the
   three training seasons (their own 2026-27 matches so far are too few to
   fit a reliable rating directly, so they aren't used for this).
5. **Fixture list** — the real official 2026-27 schedule
   (`fixtures_2026_27_raw.json`), not a simulated round-robin. Cross-checked
   against actual recorded 2026-27 results for gameweeks 1-2 (exact match).
6. **Match predictions** (`model.py`) — for each fixture, a Poisson scoreline
   grid (with the Dixon-Coles low-score adjustment), the most likely exact
   score, and Win/Draw/Loss probabilities.
7. **Season projection** (`simulate_season.py`) — Monte Carlo simulation of
   the whole season (20,000 runs by default), sampling one scoreline per
   fixture per run from that fixture's own probability grid, aggregated into
   expected points, expected final position, and title/top-4/relegation
   odds per team.

## Usage

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python model.py              # -> predictions_2026_2027.csv
python simulate_season.py    # -> projected_table_2026_2027.csv
```

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
python backtest.py 2526          # train on 2223/2324/2425, test against real 2025-26
python backtest.py 2526 3 compare        # xG-fit vs goals-fit, same target
python backtest.py 2526 3 sweep-xi       # scan decay rates, report which scored best
python backtest.py 2526 5 sweep-l2 0.002 # scan attack/defense shrinkage strength
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
- Fitting on xG vs. actual goals is close to a wash empirically: xG edged
  out actual-goals on exact-score and result accuracy in 2 of 3 seasons,
  while actual-goals was marginally better calibrated (log loss/Brier) in
  all 3. xG was kept as the default anyway since it's less exposed to
  one-off finishing luck, which matters more for a full-season simulation
  than for single-match calibration — but this is a real trade-off, not a
  clear win, and worth re-checking if the model is changed further.
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
- Re-run `sweep-xi`, `sweep-l2`, and `compare` against a season once it's
  actually completed (e.g. re-validate against 2026-27 next summer) rather
  than assuming these findings hold indefinitely — the right hyperparameters
  are a property of the data window, not universal constants.

## Interactive view

`site/index.html` has three tabs: Fixtures (all 380 matches by gameweek,
searchable), Projected Table (the season simulation), and Track Record
(pre-season predictions vs. actual results for matches played so far —
built from `track_record.py`). It reads `site/predictions.json`,
`site/table.json`, and `site/track_record.json`; regenerate those from the
CSVs after re-running the model or track_record.py — see the export
snippet in the commit that added the projected-table view, or ask for it
again if needed.

## Notes

- Data sources: football-data.co.uk (results, current-season team list) and
  Understat.com (xG, via its same-origin JSON endpoint — see `understat.py`
  for details; no login or paid access involved).
- To improve accuracy further: re-run the pipeline periodically during the
  season so ratings reflect current form rather than only the training
  window; consider a proper Bayesian shrinkage of promoted-team ratings
  toward the baseline as their own 2026-27 sample size grows, instead of the
  current all-or-nothing switch; re-tune ξ (and re-check the xG-vs-goals
  choice) with `backtest.py` periodically rather than treating either as
  fixed.
