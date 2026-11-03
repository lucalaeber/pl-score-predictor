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

import json
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://understat.com"
HEADERS = {"User-Agent": "Mozilla/5.0"}
TIMEOUT = (10, 30)  # (connect, read) seconds
RETRIES = 2
CACHE_DIR = Path(__file__).parent / "data_cache"

# Understat -> football-data.co.uk team-name spelling
TEAM_MAP = {
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton",
    "Burnley": "Burnley",
    "Cardiff": "Cardiff",
    "Chelsea": "Chelsea",
    "Coventry": "Coventry",
    "Crystal Palace": "Crystal Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Huddersfield": "Huddersfield",
    "Hull": "Hull",
    "Ipswich": "Ipswich",
    "Leeds": "Leeds",
    "Leicester": "Leicester",
    "Liverpool": "Liverpool",
    "Luton": "Luton",
    "Norwich": "Norwich",
    "Watford": "Watford",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Sheffield United": "Sheffield United",
    "Southampton": "Southampton",
    "Sunderland": "Sunderland",
    "Tottenham": "Tottenham",
    "West Bromwich Albion": "West Brom",
    "West Ham": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
}


def _download_payload(season: int) -> dict:
    """One network round trip with explicit timeouts and a couple of retries,
    so an unreachable Understat fails fast instead of hanging forever."""
    last_error = None
    for attempt in range(RETRIES + 1):
        try:
            session = requests.Session()
            session.headers.update(HEADERS)
            session.get(f"{BASE}/league/EPL/{season}", timeout=TIMEOUT)  # session cookie
            resp = session.get(
                f"{BASE}/getLeagueData/EPL/{season}",
                headers={
                    "X-Requested-With": "XMLHttpRequest",
                    "Referer": f"{BASE}/league/EPL/{season}",
                },
                timeout=TIMEOUT,
            )
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            last_error = e
            if attempt < RETRIES:
                time.sleep(2 * (attempt + 1))
    raise ConnectionError(f"Could not reach Understat for season {season}: {last_error}")


def fetch_understat_season(season: int) -> pd.DataFrame:
    """Return every EPL match of the given season (e.g. 2023 = 2023-24)
    with columns: Date, HomeTeam, AwayTeam, FTHG, FTAG, HxG, AxG, Played.
    Unplayed fixtures have NaN goals/xG.

    A fully played season never changes, so once fetched it's cached on disk
    (data_cache/) and never requested again. An in-progress season is
    re-fetched every time, but if Understat is unreachable the last cached
    copy is used (with a warning) rather than failing the whole run."""
    cache_path = CACHE_DIR / f"understat_EPL_{season}.json"
    cached = json.loads(cache_path.read_text()) if cache_path.exists() else None

    if cached and cached["complete"]:
        rows = cached["rows"]
    else:
        try:
            rows = _parse_payload(_download_payload(season))
            CACHE_DIR.mkdir(exist_ok=True)
            cache_path.write_text(json.dumps({
                "complete": all(r["Played"] for r in rows),
                "rows": rows,
            }))
        except ConnectionError as e:
            if not cached:
                raise
            print(f"  warning: {e}\n  using cached Understat data for {season} "
                  f"(may be missing recent matches)")
            rows = cached["rows"]

    df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def _parse_payload(payload: dict) -> list:
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
    return rows
