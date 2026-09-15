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
   likelihood, with exponential time decay (ξ = 0.0065/day) so recent
   results count more. Rho is still fit against actual scorelines, since the
   effect it corrects for (0-0/1-0/0-1/1-1 being more or less common than
   independent Poisson predicts) is a real discrete-score phenomenon that xG
   wouldn't show.
4. **Promoted-team baseline** — teams with no recent top-flight history are
   assigned the average rating of the team that finished 18th in each of the
   three training seasons (their own 2026-27 matches so far are too few to
   fit a reliable rating directly, so they aren't used for this).
4. **Fixture list** — the real official 2026-27 schedule (`fixtures_2026_27_raw.json`),
   not a simulated round-robin. Cross-checked against actual recorded
   2026-27 results for gameweeks 1-2 (exact match).
5. **Match predictions** (`model.py`) — for each fixture, a Poisson scoreline
   grid (with the Dixon-Coles low-score adjustment), the most likely exact
   score, and Win/Draw/Loss probabilities.
6. **Season projection** (`simulate_season.py`) — Monte Carlo simulation of
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

## Interactive view

`site/index.html` is a browsable page (fixtures by gameweek, searchable, plus
the projected table) built from `site/predictions.json` and `site/table.json`.
Regenerate those from the CSVs after re-running the model, e.g.:

```bash
python - <<'EOF'
import pandas as pd, json
# see conversation history / commit log for the exact export snippet
EOF
```

## Notes

- Data sources: football-data.co.uk (results, current-season team list) and
  Understat.com (xG, via its same-origin JSON endpoint — see `understat.py`
  for details; no login or paid access involved).
- To improve accuracy further: re-run the pipeline periodically during the
  season so ratings reflect current form rather than only the 2023-26
  training window; consider a proper Bayesian shrinkage of promoted-team
  ratings toward the baseline as their own 2026-27 sample size grows,
  instead of the current all-or-nothing switch.
