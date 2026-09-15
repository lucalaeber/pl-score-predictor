"""
Understat.com expected-goals (xG) data access.

Understat has no official public API, but the season page's own frontend
calls a same-origin JSON endpoint (`getLeagueData/<league>/<season>`) to
render its tables. This mirrors that call: load the season page once to
pick up its session cookie, then request the JSON directly. That endpoint
returns every match of the season (played and upcoming) with home/away
team names, actual goals, and each side's xG for that specific match.

No login or paid access is involved -- this is the same data Understat
already serves to any visitor's browser, just read directly as JSON
instead of rendered into a webpage.
"""

import pandas as pd
import requests

BASE = "https://understat.com"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Understat -> football-data.co.uk team-name spelling
TEAM_MAP = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Burnley": "Burnley",
    "Chelsea": "Chelsea",
    "Coventry": "Coventry",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Hull": "Hull",
    "Ipswich": "Ipswich",
    "Leeds": "Leeds",
    "Leicester": "Leicester",
    "Liverpool": "Liverpool",
    "Luton": "Luton",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Sheffield United": "Sheffield United",
    "Southampton": "Southampton",
    "Sunderland": "Sunderland",
    "Tottenham": "Tottenham",
    "West Ham": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
}


def fetch_understat_season(season: int) -> pd.DataFrame:
    """Return every EPL match of the given season (e.g. 2023 = 2023-24)
    with columns: Date, HomeTeam, AwayTeam, FTHG, FTAG, HxG, AxG, Played.
    Unplayed fixtures have NaN goals/xG."""
    session = requests.Session()
    session.headers.update(HEADERS)
    session.get(f"{BASE}/league/EPL/{season}")  # picks up session cookie

    resp = session.get(
        f"{BASE}/getLeagueData/EPL/{season}",
        headers={
            "X-Requested-With": "XMLHttpRequest",
            "Referer": f"{BASE}/league/EPL/{season}",
        },
    )
    resp.raise_for_status()
    payload = resp.json()

    rows = []
    for m in payload["dates"]:
        home_raw, away_raw = m["h"]["title"], m["a"]["title"]
        if home_raw not in TEAM_MAP or away_raw not in TEAM_MAP:
            raise ValueError(f"Unmapped Understat team name: {home_raw!r} / {away_raw!r}")
        rows.append({
            "Date": m["datetime"],
            "HomeTeam": TEAM_MAP[home_raw],
            "AwayTeam": TEAM_MAP[away_raw],
            "FTHG": int(m["goals"]["h"]) if m["isResult"] else None,
            "FTAG": int(m["goals"]["a"]) if m["isResult"] else None,
            "HxG": float(m["xG"]["h"]) if m["isResult"] else None,
            "AxG": float(m["xG"]["a"]) if m["isResult"] else None,
            "Played": m["isResult"],
        })

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    return df
