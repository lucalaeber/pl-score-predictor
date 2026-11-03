"""
How big is the injury effect, according to this season's own data?

For every finished 2026-27 match, work out how much of each team's attacking
output was actually absent (players with a big share of the team's xG + xA
who played few or no minutes), and compare the team's real xG in that match
with what a pre-season model expected. The slope of "relative xG" against
"missing share" is the effect injuries.py assumes (SENSITIVITY).

Caveats that matter:
  * One season, ~8 gameweeks: the estimate is very noisy. The point of this
    script is to check the assumption isn't wildly off, not to fit it.
  * Importance shares use season-to-date totals excluding the match being
    scored, so a player's own output in that match can't inflate his share.
  * "Missing" is minutes-based (a player subbed off early counts partly), so
    it captures tactical rotation as well as injuries.

Run:
    python injury_effect.py
"""

import collections

import numpy as np
import pandas as pd

from model import DixonColes, XI, TRAIN_SEASON_CODES, load_season, eighteenth_place_baseline
from injuries import MIN_SHARE, _get, fetch_fpl, sensitivity, team_names

FULL_MATCH_MINUTES = 72  # a regular playing 80% of a match counts as fully available


def preseason_model(new_teams_hint=None):
    train = pd.concat([load_season(c) for c in TRAIN_SEASON_CODES], ignore_index=True)
    teams = sorted(set(train["HomeTeam"]) | set(train["AwayTeam"]))
    model = DixonColes(teams)
    model.fit(train, as_of=train["Date"].max() + pd.Timedelta(days=1), xi=XI, use_xg=False)
    return model, train


def main():
    bootstrap, fixtures = fetch_fpl()
    names = team_names(bootstrap)
    player_team = {p["id"]: p["team"] for p in bootstrap["elements"]}
    finished = sorted({f["event"] for f in fixtures if f["finished"]})
    print(f"Finished gameweeks: {finished}")

    # per gameweek, per player: minutes, xG, xGI
    live = {}
    for g in finished:
        live[g] = {e["id"]: e["stats"] for e in _get(f"event/{g}/live/")["elements"]}

    total_xgi = collections.defaultdict(float)       # player -> season xGI
    team_total = collections.defaultdict(float)
    for g in finished:
        for pid, s in live[g].items():
            x = float(s["expected_goal_involvements"])
            total_xgi[pid] += x
            team_total[player_team[pid]] += x

    model, train = preseason_model()
    model_teams = set(model.teams)
    new_teams = sorted({names[t] for t in names} - model_teams)
    if new_teams:
        ba, bd = eighteenth_place_baseline(train, model)
        for t in new_teams:
            model.params_["att"][t], model.params_["def"][t] = ba, bd
            model.teams.append(t)

    fixtures_per_team = collections.Counter()
    for f in fixtures:
        if f["finished"]:
            fixtures_per_team[(f["event"], f["team_h"])] += 1
            fixtures_per_team[(f["event"], f["team_a"])] += 1

    rows = []
    for f in fixtures:
        if not f["finished"]:
            continue
        g = f["event"]
        expected = model.score_matrix(names[f["team_h"]], names[f["team_a"]])
        for team, exp_goals in ((f["team_h"], expected[1]), (f["team_a"], expected[2])):
            if fixtures_per_team[(g, team)] != 1:
                continue  # double gameweek: live stats mix two matches
            squad = [pid for pid, t in player_team.items() if t == team]
            gw_team_xgi = sum(float(live[g][p]["expected_goal_involvements"]) for p in squad if p in live[g])
            denom = team_total[team] - gw_team_xgi
            if denom <= 0:
                continue
            missing = 0.0          # minutes-based: includes routine substitutions
            absent = 0.0           # strict: regulars who barely played
            for p in squad:
                s = live[g].get(p)
                own = float(s["expected_goal_involvements"]) if s else 0.0
                share = (total_xgi[p] - own) / denom
                if share < MIN_SHARE:
                    continue
                minutes = s["minutes"] if s else 0
                missing += share * max(0.0, 1 - min(1.0, minutes / FULL_MATCH_MINUTES))
                other = [live[e][p]["minutes"] if p in live[e] else 0 for e in finished if e != g]
                if other and np.mean(other) >= 50 and minutes < 15:
                    absent += share
            actual_xg = sum(float(live[g][p]["expected_goals"]) for p in squad if p in live[g])
            rows.append({"gw": g, "team": names[team], "missing": missing, "absent": absent,
                         "rel_xg": actual_xg / exp_goals})

    df = pd.DataFrame(rows)

    def beta(frame, col):
        b, a = np.polyfit(frame[col], frame["rel_xg"], 1)
        return -b / a

    rng = np.random.default_rng(1)
    assumed = sensitivity(bootstrap)
    print(f"\n{len(df)} team-matches. Effect on a team's xG per 10% of its xG+xA share lost")
    print(f"(injuries.py assumes {assumed * 10:.1f}%: the substitute replaces half of a missing "
          f"player's output, and xG+xA counts each chance twice)\n")
    for col, label in (("missing", "minutes-based (includes substitutions)"),
                       ("absent", "strict (regulars who barely played)")):
        est = beta(df, col)
        boots = [beta(df.sample(len(df), replace=True, random_state=int(rng.integers(1e9))), col)
                 for _ in range(2000)]
        lo, hi = np.percentile(boots, [5, 95])
        print(f"  {label}")
        print(f"    mean share lost {df[col].mean():.1%}; matches with >10% lost: {(df[col] > 0.10).mean():.0%}")
        print(f"    xG falls by {est * 10:.1f}% per 10% lost   (90% interval {lo * 10:.1f}% to {hi * 10:.1f}%)"
              f"  -> assumption {'inside' if lo <= assumed <= hi else 'OUTSIDE'} the interval\n")


if __name__ == "__main__":
    main()
