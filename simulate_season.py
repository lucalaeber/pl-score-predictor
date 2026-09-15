"""
Full-season Monte Carlo projection.

Uses the fitted Dixon-Coles model (see model.py) to simulate the entire
2026-27 season thousands of times -- sampling one scoreline per fixture from
that fixture's own score-probability grid each time -- then aggregates the
resulting final tables into per-team projections: expected points, expected
final position, and the probability of finishing in the title, top-4
(Champions League qualification), or bottom-3 (relegation) spots.

Run:
    python simulate_season.py [n_simulations]

Output:
    projected_table_2026_2027.csv
"""

import sys

import numpy as np
import pandas as pd

from model import build_model_and_fixtures

N_SIMS_DEFAULT = 20000
RELEGATION_SPOTS = 3
EUROPE_SPOTS = 4  # Champions League qualification places


def simulate_season(model, fixtures: pd.DataFrame, n_sims: int, seed: int = 7):
    rng = np.random.default_rng(seed)

    teams = sorted(set(fixtures["HomeTeam"]) | set(fixtures["AwayTeam"]))
    team_idx = {t: i for i, t in enumerate(teams)}
    n_teams = len(teams)
    n_fixtures = len(fixtures)

    home_idx = fixtures["HomeTeam"].map(team_idx).to_numpy()
    away_idx = fixtures["AwayTeam"].map(team_idx).to_numpy()

    pts = np.zeros((n_sims, n_teams), dtype=np.int32)
    gf = np.zeros((n_sims, n_teams), dtype=np.int32)
    ga = np.zeros((n_sims, n_teams), dtype=np.int32)

    max_g = model.score_matrix(fixtures.iloc[0]["HomeTeam"], fixtures.iloc[0]["AwayTeam"])[0].shape[0]

    for i in range(n_fixtures):
        home, away = fixtures.iloc[i]["HomeTeam"], fixtures.iloc[i]["AwayTeam"]
        matrix, _, _ = model.score_matrix(home, away)
        flat_probs = matrix.flatten()
        flat_probs = flat_probs / flat_probs.sum()

        draws = rng.choice(len(flat_probs), size=n_sims, p=flat_probs)
        hg = draws // max_g
        ag = draws % max_g

        h, a = home_idx[i], away_idx[i]
        home_win = hg > ag
        away_win = hg < ag
        draw = ~home_win & ~away_win

        pts[:, h] += np.where(home_win, 3, np.where(draw, 1, 0))
        pts[:, a] += np.where(away_win, 3, np.where(draw, 1, 0))
        gf[:, h] += hg
        ga[:, h] += ag
        gf[:, a] += ag
        ga[:, a] += hg

    gd = gf - ga
    # Composite sort key: points dominate, then goal difference, then goals
    # for. Coefficients keep each component from overflowing into the next
    # given realistic PL score ranges.
    sort_key = pts.astype(np.float64) * 1_000 + gd * 1.0 + gf * 0.001
    order = np.argsort(-sort_key, axis=1)  # best-to-worst team index per sim
    position = np.empty_like(order)
    sim_rows = np.arange(n_sims)[:, None]
    position[sim_rows, order] = np.arange(1, n_teams + 1)[None, :]

    table = pd.DataFrame({
        "Team": teams,
        "Expected_Points": pts.mean(axis=0).round(1),
        "Expected_GD": gd.mean(axis=0).round(1),
        "Expected_Position": position.mean(axis=0).round(2),
        "P_Title": (position == 1).mean(axis=0),
        "P_Top4": (position <= EUROPE_SPOTS).mean(axis=0),
        "P_Relegation": (position > n_teams - RELEGATION_SPOTS).mean(axis=0),
    })
    table = table.sort_values("Expected_Position").reset_index(drop=True)
    table.insert(0, "Projected_Rank", np.arange(1, n_teams + 1))
    for col in ("P_Title", "P_Top4", "P_Relegation"):
        table[col] = table[col].round(4)
    return table


def main():
    n_sims = int(sys.argv[1]) if len(sys.argv) > 1 else N_SIMS_DEFAULT

    model, fixtures, new_teams = build_model_and_fixtures()

    print(f"Simulating the season {n_sims} times...")
    table = simulate_season(model, fixtures, n_sims)
    if new_teams:
        table["Promoted_No_PL_History"] = table["Team"].isin(new_teams)

    out_path = "projected_table_2026_2027.csv"
    table.to_csv(out_path, index=False)
    print(f"Wrote projected final table to {out_path}\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
