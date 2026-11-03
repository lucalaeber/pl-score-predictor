"""
Injury / availability adjustment, from the Fantasy Premier League API.

The FPL API is free, public JSON (no login) and carries, for every player,
a live status (available / injured / doubtful / suspended / unavailable), the
chance of playing the next round, a news line, and season-to-date expected
goals and assists. From that this module works out, per team, how much of its
attacking output is expected to be missing from an upcoming fixture.

What is and isn't modelled -- worth knowing before trusting the numbers:
  * Attack only. A player's importance is his share of his team's xG + xA
    this season. There's no equally good free signal for defensive importance,
    so missing defenders and goalkeepers are NOT adjusted for.
  * The strength of the effect is an ASSUMPTION, not a fitted value:
    REPLACEMENT_SHARE is how much of a missing player's output the substitute
    supplies. injury_effect.py checks it against this season's matches; the
    sample is small (~180 team-matches), so it can only confirm the assumption
    is in a plausible range, not pin it down. FPL has no historical injury
    snapshots, so unlike the rest of the model this can't be backtested over
    past seasons.
  * It's a snapshot of the news as of the moment it's fetched, so it only
    makes sense for the next gameweek's fixtures.
"""

import collections

import requests

FPL = "https://fantasy.premierleague.com/api"
HEADERS = {"User-Agent": "Mozilla/5.0"}
TIMEOUT = (10, 30)

# FPL team name -> football-data.co.uk spelling used throughout the project
FPL_TEAM_MAP = {
    "Arsenal": "Arsenal", "Aston Villa": "Aston Villa", "Bournemouth": "Bournemouth",
    "Brentford": "Brentford", "Brighton": "Brighton", "Chelsea": "Chelsea",
    "Coventry City": "Coventry", "Crystal Palace": "Crystal Palace", "Everton": "Everton",
    "Fulham": "Fulham", "Hull City": "Hull", "Ipswich Town": "Ipswich", "Leeds": "Leeds",
    "Liverpool": "Liverpool", "Man City": "Man City", "Man Utd": "Man United",
    "Newcastle": "Newcastle", "Nott'm Forest": "Nott'm Forest", "Spurs": "Tottenham",
    "Sunderland": "Sunderland",
}

REPLACEMENT_SHARE = 0.5   # assumption: substitute supplies half of a missing player's output
MAX_MISSING_SHARE = 0.5   # cap on a team's missing share
MIN_GOAL_SCALE = 0.5      # a team's expected goals are never scaled below half
MIN_SHARE = 0.02          # ignore players below 2% of team xG+xA


def _get(path):
    resp = requests.get(f"{FPL}/{path}", headers=HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def fetch_fpl():
    return _get("bootstrap-static/"), _get("fixtures/")


def team_names(bootstrap):
    names = {}
    for t in bootstrap["teams"]:
        if t["name"] not in FPL_TEAM_MAP:
            raise ValueError(f"Unmapped FPL team name: {t['name']!r}")
        names[t["id"]] = FPL_TEAM_MAP[t["name"]]
    return names


def target_event(bootstrap, fixtures):
    """The gameweek whose fixtures we're predicting: the earliest one that
    still has unplayed matches, and whether it's the one currently running
    (which decides whether `this round` or `next round` chance applies)."""
    unfinished = [f for f in fixtures if not f["finished"] and f["event"] is not None]
    event = min(f["event"] for f in unfinished)
    current = next((e["id"] for e in bootstrap["events"] if e["is_current"]), None)
    return event, event == current


def p_out(player, use_this_round):
    """Probability the player misses the target gameweek."""
    chance = player["chance_of_playing_this_round" if use_this_round else "chance_of_playing_next_round"]
    if chance is not None:
        return 1 - chance / 100
    return 0.0 if player["status"] == "a" else 1.0


def missing_attack(bootstrap, use_this_round):
    """{team name: {"share": missing share of team xG+xA, "players": [...]}}"""
    names = team_names(bootstrap)
    total = collections.defaultdict(float)
    for p in bootstrap["elements"]:
        total[p["team"]] += float(p["expected_goal_involvements"])

    out = {name: {"share": 0.0, "players": []} for name in names.values()}
    for p in bootstrap["elements"]:
        if not total[p["team"]]:
            continue
        share = float(p["expected_goal_involvements"]) / total[p["team"]]
        po = p_out(p, use_this_round)
        if share < MIN_SHARE or po <= 0:
            continue
        team = out[names[p["team"]]]
        team["share"] += share * po
        team["players"].append({
            "name": p["web_name"], "share": round(share, 3), "p_out": round(po, 2),
            "status": p["status"], "news": p["news"],
        })
    for team in out.values():
        team["share"] = min(team["share"], MAX_MISSING_SHARE)
        team["players"].sort(key=lambda x: -x["share"] * x["p_out"])
    return out


def involvement_factor(bootstrap):
    """League-wide xG+xA divided by xG. A chance is counted once for the
    shooter (xG) and once for the creator (xA), so a player's share of team
    xG+xA overstates the share of the team's chances he's involved in by
    roughly this factor (~1.75); it's measured from the data, not assumed."""
    xg = sum(float(p["expected_goals"]) for p in bootstrap["elements"])
    xgi = sum(float(p["expected_goal_involvements"]) for p in bootstrap["elements"])
    return xgi / xg


def sensitivity(bootstrap):
    """Fraction of a team's xG lost per unit of missing xG+xA share: the
    chances the missing player is involved in, times the share of that
    output the substitute doesn't replace."""
    return (1 - REPLACEMENT_SHARE) * involvement_factor(bootstrap)


def goal_scale(missing_share, sens):
    """Multiplier on a team's expected goals given its missing attacking share."""
    return max(MIN_GOAL_SCALE, 1 - sens * missing_share)


def upcoming_fixtures(bootstrap, fixtures):
    """Fixtures of the target gameweek with each side's missing attacking
    share and key absentees, ready to feed into score_matrix(lam_scale=...)."""
    event, use_this_round = target_event(bootstrap, fixtures)
    names = team_names(bootstrap)
    missing = missing_attack(bootstrap, use_this_round)
    sens = sensitivity(bootstrap)
    rows = []
    for f in sorted((f for f in fixtures if f["event"] == event and not f["finished"]),
                    key=lambda f: f["kickoff_time"] or ""):
        home, away = names[f["team_h"]], names[f["team_a"]]
        rows.append({
            "event": event, "kickoff": f["kickoff_time"], "home": home, "away": away,
            "home_missing": missing[home]["share"], "away_missing": missing[away]["share"],
            "home_scale": goal_scale(missing[home]["share"], sens),
            "away_scale": goal_scale(missing[away]["share"], sens),
            "home_players": missing[home]["players"][:3], "away_players": missing[away]["players"][:3],
            "sensitivity": sens,
        })
    return rows
