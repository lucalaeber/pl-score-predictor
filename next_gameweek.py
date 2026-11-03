"""
Next-gameweek predictions, with and without the injury adjustment.

Fetches live availability from the FPL API (see injuries.py), then for each
fixture of the upcoming gameweek computes the model's prediction twice: as
normal, and with each side's expected goals scaled down by how much of its
attacking output is expected to be missing.

Run:
    python next_gameweek.py                # full model (needs Understat)
    python next_gameweek.py --goals-only   # development only, not publishable

Output:
    next_gameweek.csv
    site/next_gameweek.json
"""

import datetime as dt
import json
import sys

import numpy as np
import pandas as pd

from model import build_model_and_fixtures
from injuries import REPLACEMENT_SHARE, fetch_fpl, upcoming_fixtures
from teams import CODES


def summarise(matrix, lam, mu):
    h, a = np.unravel_index(np.argmax(matrix), matrix.shape)
    return {
        "score": f"{h}-{a}",
        "xh": round(float(lam), 2), "xa": round(float(mu), 2),
        "ph": round(float(np.tril(matrix, -1).sum()) * 100, 1),
        "pd": round(float(np.trace(matrix)) * 100, 1),
        "pa": round(float(np.triu(matrix, 1).sum()) * 100, 1),
    }


def main():
    goals_only = "--goals-only" in sys.argv
    model, _, _ = build_model_and_fixtures(goals_only=goals_only)

    print("\nFetching live availability from the FPL API...")
    bootstrap, fpl_fixtures = fetch_fpl()
    upcoming = upcoming_fixtures(bootstrap, fpl_fixtures)
    event = upcoming[0]["event"]
    print(f"Gameweek {event}: {len(upcoming)} fixtures\n")

    rows = []
    for f in upcoming:
        base = summarise(*model.score_matrix(f["home"], f["away"]))
        adj = summarise(*model.score_matrix(
            f["home"], f["away"], lam_scale=f["home_scale"], mu_scale=f["away_scale"],
        ))
        rows.append({
            "kickoff": f["kickoff"], "home": f["home"], "away": f["away"],
            "hc": CODES[f["home"]], "ac": CODES[f["away"]],
            "home_missing": round(f["home_missing"] * 100, 1),
            "away_missing": round(f["away_missing"] * 100, 1),
            "home_players": f["home_players"], "away_players": f["away_players"],
            "base": base, "adjusted": adj,
        })
        moved = adj["ph"] - base["ph"]
        print(f"{f['home']:<15} v {f['away']:<15} missing {f['home_missing']:>4.0%} / {f['away_missing']:>4.0%}"
              f"   home win {base['ph']:>5.1f}% -> {adj['ph']:>5.1f}% ({moved:+.1f})   "
              f"score {base['score']} -> {adj['score']}")

    pd.DataFrame([{
        "Gameweek": event, "Kickoff": r["kickoff"], "HomeTeam": r["home"], "AwayTeam": r["away"],
        "Home_Missing_Attack_Pct": r["home_missing"], "Away_Missing_Attack_Pct": r["away_missing"],
        "Base_Score": r["base"]["score"], "Adjusted_Score": r["adjusted"]["score"],
        "Base_P_Home": r["base"]["ph"], "Adjusted_P_Home": r["adjusted"]["ph"],
        "Base_P_Draw": r["base"]["pd"], "Adjusted_P_Draw": r["adjusted"]["pd"],
        "Base_P_Away": r["base"]["pa"], "Adjusted_P_Away": r["adjusted"]["pa"],
    } for r in rows]).to_csv("next_gameweek.csv", index=False)

    payload = {
        "generated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="minutes"),
        "event": event,
        "mode": "goals-only (development)" if goals_only else "xG + goals ensemble",
        "assumption": {"replacement_share": REPLACEMENT_SHARE,
                       "sensitivity": round(upcoming[0]["sensitivity"], 3)},
        "fixtures": rows,
    }
    with open("site/next_gameweek.json", "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print("\nWrote next_gameweek.csv and site/next_gameweek.json")


if __name__ == "__main__":
    main()
