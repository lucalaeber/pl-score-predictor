"""
Track record: how would the model's predictions have looked *before* the
2026-27 season started, checked against what's actually happened so far?

This reuses backtest.py's machinery but points it at the in-progress
2026-27 season. Crucially, it fits on ONLY the prior completed seasons
(see model.py's TRAIN_SEASON_CODES) -- unlike predictions_2026_2027.csv's
model, which folds already-played 2026-27 matches into training for
rating purposes and so would be "cheating" if used to grade those same
matches (it already saw the results). This gives an honest, un-leaked
pre-season prediction for every match played so far, to check against
what actually happened.

Run:
    python track_record.py

Output:
    track_record_2026_2027.csv
    site/track_record.json
"""

import json

import numpy as np
import pandas as pd

from model import load_real_fixtures
from backtest import run_backtest, compute_match_metrics

TARGET_CODE = "2627"
N_TRAIN = 5  # keep in sync with model.py's TRAIN_SEASON_CODES length

CODES = {
    "Arsenal": "ARS", "Aston Villa": "AVL", "Bournemouth": "BOU", "Brentford": "BRE",
    "Brighton": "BHA", "Chelsea": "CHE", "Coventry": "COV", "Crystal Palace": "CRY",
    "Everton": "EVE", "Fulham": "FUL", "Hull": "HUL", "Ipswich": "IPS", "Leeds": "LEE",
    "Liverpool": "LIV", "Man City": "MCI", "Man United": "MUN", "Newcastle": "NEW",
    "Nott'm Forest": "NFO", "Sunderland": "SUN", "Tottenham": "TOT",
}


def main():
    print(f"Fitting a pre-season-only model (trained on the {N_TRAIN} seasons before {TARGET_CODE},")
    print("no 2026-27 results included) and checking it against matches played so far...\n")

    _, results, training_data, _ = run_backtest(TARGET_CODE, N_TRAIN, use_xg=True)
    print(f"\n{len(results)} matches played so far in 2026-27\n")

    m = compute_match_metrics(results, training_data)
    print(f"{'Metric':<28}{'Model':>10}{'Naive baseline':>18}")
    print("-" * 56)
    print(f"{'Exact scoreline accuracy':<28}{m['exact_acc']:>9.1%} {m['naive_exact_acc']:>17.1%}")
    print(f"{'Match result accuracy':<28}{m['result_acc']:>9.1%} {m['naive_result_acc']:>17.1%}")
    print(f"{'Log loss (lower better)':<28}{m['log_loss']:>10.4f}{m['naive_log_loss']:>19.4f}")
    print(f"{'Brier score (lower better)':<28}{m['brier']:>10.4f}{m['naive_brier']:>19.4f}")

    # attach real gameweek numbers from the official fixture list
    fixtures = load_real_fixtures()[["Gameweek", "HomeTeam", "AwayTeam"]]
    results = results.merge(fixtures, on=["HomeTeam", "AwayTeam"], how="left")
    results = results.sort_values(["Gameweek", "Date"]).reset_index(drop=True)

    out_path = "track_record_2026_2027.csv"
    results.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    records = []
    for _, r in results.iterrows():
        records.append({
            "gw": int(r["Gameweek"]),
            "date": r["Date"],
            "h": r["HomeTeam"], "a": r["AwayTeam"],
            "hc": CODES[r["HomeTeam"]], "ac": CODES[r["AwayTeam"]],
            "actual": r["ActualScore"],
            "pred": r["PredictedScore"],
            "exact_hit": bool(r["ActualScore"] == r["PredictedScore"]),
            "result_hit": bool(r["ActualResult"] == r["PredictedResult"]),
            "ph": round(float(r["P_Home"]) * 100, 1),
            "pd": round(float(r["P_Draw"]) * 100, 1),
            "pa": round(float(r["P_Away"]) * 100, 1),
        })
    summary = {
        "n": int(m["n"]),
        "exact_acc": round(float(m["exact_acc"]) * 100, 1),
        "result_acc": round(float(m["result_acc"]) * 100, 1),
        "naive_result_acc": round(float(m["naive_result_acc"]) * 100, 1),
        "log_loss": round(float(m["log_loss"]), 4),
        "naive_log_loss": round(float(m["naive_log_loss"]), 4),
        "gws_covered": sorted(set(r["gw"] for r in records)),
    }
    with open("site/track_record.json", "w") as f:
        json.dump({"summary": summary, "matches": records}, f, separators=(",", ":"))
    print("Wrote site/track_record.json")


if __name__ == "__main__":
    main()
