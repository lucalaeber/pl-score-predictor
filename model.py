"""
Static Premier League Score Predictor
======================================
Dixon-Coles maximum-likelihood attack/defense model fit on the last three
completed Premier League seasons (2023-24, 2024-25, 2025-26), used to predict
the most probable scoreline and outcome probabilities for all 380 fixtures
of the 2026-27 season.

No external APIs beyond plain CSV downloads from football-data.co.uk (pandas
reads the CSV URLs directly). Everything else -- fitting, simulation,
fixture generation -- runs locally.

Run:
    python model.py

Output:
    predictions_2026_2027.csv
"""

import datetime as dt
import math

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from understat import fetch_understat_season

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

TRAIN_SEASON_CODES = ["2324", "2425", "2526"]  # 2023-24, 2024-25, 2025-26
CURRENT_SEASON_CODE = "2627"                    # 2026-27 (in progress)
BASE_URL = "https://www.football-data.co.uk/mmz4281/{code}/E0.csv"

XI = 0.0065          # exponential time-decay rate (per day)
MAX_GOALS = 6         # truncate scoreline grid at 6-6

COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]


# ---------------------------------------------------------------------------
# 1. Data ingestion
# ---------------------------------------------------------------------------

def load_season(code: str, want_xg: bool = False) -> pd.DataFrame:
    url = BASE_URL.format(code=code)
    raw = pd.read_csv(url)
    cols = COLS + ["HxG", "AxG"] if want_xg and "HxG" in raw.columns else COLS
    df = raw[cols].dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Date"])
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    df["SeasonCode"] = code
    if "HxG" not in df.columns:
        df["HxG"] = np.nan
        df["AxG"] = np.nan
    return df


def understat_year_for_code(code: str) -> int:
    return int("20" + code[:2])


def attach_understat_xg(df: pd.DataFrame, code: str) -> pd.DataFrame:
    """Merge in Understat's per-match xG for both sides, matched on the
    (HomeTeam, AwayTeam) pair -- each pair occurs at most once per season in
    a given venue order, so no date join is needed (and Understat's kickoff
    timestamps aren't guaranteed to line up with football-data.co.uk's)."""
    xg = fetch_understat_season(understat_year_for_code(code))
    xg = xg[xg["Played"]][["HomeTeam", "AwayTeam", "HxG", "AxG"]]
    merged = df.drop(columns=["HxG", "AxG"]).merge(
        xg, on=["HomeTeam", "AwayTeam"], how="left"
    )
    missing = merged["HxG"].isna().sum()
    if missing:
        print(f"  warning: {missing}/{len(merged)} matches in {code} have no Understat xG match")
    return merged


def load_training_data() -> pd.DataFrame:
    frames = []
    for code in TRAIN_SEASON_CODES:
        season_df = load_season(code)
        season_df = attach_understat_xg(season_df, code)
        frames.append(season_df)

    historical_teams = set()
    for f in frames:
        historical_teams |= set(f["HomeTeam"]) | set(f["AwayTeam"])

    # Also train on the current (in-progress) season's played matches so far
    # -- football-data.co.uk already publishes HxG/AxG for it directly, and
    # the exponential time decay in DixonColes.fit will naturally weight
    # these most heavily since they're the most recent matches available.
    # Matches involving a team with no 2023-26 history (a fresh promotion)
    # are excluded here: 3-4 games isn't enough to fit that team's own
    # rating reliably, so it keeps the 18th-place historical baseline
    # instead (see eighteenth_place_baseline / build_model_and_fixtures).
    current_played = load_season(CURRENT_SEASON_CODE, want_xg=True)
    current_played = current_played[
        current_played["HomeTeam"].isin(historical_teams)
        & current_played["AwayTeam"].isin(historical_teams)
    ]
    frames.append(current_played)

    data = pd.concat(frames, ignore_index=True)
    data = data.sort_values("Date").reset_index(drop=True)
    return data


def load_current_season_teams() -> list:
    """Read the in-progress 2026-27 CSV purely to discover this season's
    20-team membership (who was promoted/relegated), independent of how
    many matches have been played so far."""
    df = load_season(CURRENT_SEASON_CODE)
    teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    if len(teams) != 20:
        raise ValueError(
            f"Expected 20 teams in {CURRENT_SEASON_CODE} season file, found "
            f"{len(teams)}: {teams}"
        )
    return teams


# ---------------------------------------------------------------------------
# 2. Season standings (used to find each season's 18th-place team, for the
#    promoted-team baseline rating)
# ---------------------------------------------------------------------------

def season_table(season_df: pd.DataFrame) -> pd.DataFrame:
    """Return final standings (Pts, GD, GF) for one season's match data,
    sorted best-to-worst (index 0 = champions, index 19 = 20th place)."""
    teams = sorted(set(season_df["HomeTeam"]) | set(season_df["AwayTeam"]))
    pts = {t: 0 for t in teams}
    gf = {t: 0 for t in teams}
    ga = {t: 0 for t in teams}

    for _, row in season_df.iterrows():
        h, a, hg, ag = row["HomeTeam"], row["AwayTeam"], row["FTHG"], row["FTAG"]
        gf[h] += hg
        ga[h] += ag
        gf[a] += ag
        ga[a] += hg
        if hg > ag:
            pts[h] += 3
        elif hg < ag:
            pts[a] += 3
        else:
            pts[h] += 1
            pts[a] += 1

    table = pd.DataFrame({
        "Team": teams,
        "Pts": [pts[t] for t in teams],
        "GD": [gf[t] - ga[t] for t in teams],
        "GF": [gf[t] for t in teams],
    })
    table = table.sort_values(["Pts", "GD", "GF"], ascending=False).reset_index(drop=True)
    return table


# ---------------------------------------------------------------------------
# 3. Dixon-Coles model
# ---------------------------------------------------------------------------

class DixonColes:
    def __init__(self, teams):
        self.teams = list(teams)
        self.n = len(self.teams)
        self.idx = {t: i for i, t in enumerate(self.teams)}
        self.params_ = None  # dict: att, def, home_adv, rho

    @staticmethod
    def _tau(x, y, lam, mu, rho):
        if x == 0 and y == 0:
            return 1 - lam * mu * rho
        elif x == 0 and y == 1:
            return 1 + lam * rho
        elif x == 1 and y == 0:
            return 1 + mu * rho
        elif x == 1 and y == 1:
            return 1 - rho
        return 1.0

    def _unpack(self, x):
        n = self.n
        att_free = x[0:n - 1]
        def_free = x[n - 1:2 * (n - 1)]
        home_adv = x[2 * (n - 1)]
        rho = x[2 * (n - 1) + 1]
        # last team's att/def fixed so that sum(att) = 0, sum(def) = 0
        att = np.append(att_free, -att_free.sum())
        deff = np.append(def_free, -def_free.sum())
        return att, deff, home_adv, rho

    def _neg_log_likelihood(self, x, home_idx, away_idx, hg_fit, ag_fit, hg_actual, ag_actual, weights):
        att, deff, home_adv, rho = self._unpack(x)

        log_lam = home_adv + att[home_idx] + deff[away_idx]
        log_mu = att[away_idx] + deff[home_idx]
        lam = np.exp(log_lam)
        mu = np.exp(log_mu)

        # Poisson-style loss (unnormalized, factorial terms dropped) driving
        # attack/defense/home-advantage. hg_fit/ag_fit is xG when available
        # (a smoother, less score-noisy target than the actual final score)
        # falling back to actual goals otherwise.
        ll = hg_fit * log_lam - lam + ag_fit * log_mu - mu

        # Dixon-Coles low-score correction: this corrects a real discrete-
        # scoreline effect (0-0/1-0/0-1/1-1 are more/less common than
        # independent Poisson predicts), so it's evaluated against the
        # *actual* final score, not xG -- xG values are essentially never
        # exactly 0 or 1, which would make the correction a no-op.
        tau = np.array([
            self._tau(x_, y_, l_, m_, rho)
            for x_, y_, l_, m_ in zip(hg_actual, ag_actual, lam, mu)
        ])
        tau = np.clip(tau, 1e-10, None)  # guard against invalid rho region
        ll = ll + np.log(tau)

        return -np.sum(weights * ll)

    def fit(self, matches: pd.DataFrame, as_of: dt.datetime, xi: float = XI):
        home_idx = matches["HomeTeam"].map(self.idx).to_numpy()
        away_idx = matches["AwayTeam"].map(self.idx).to_numpy()
        hg_actual = matches["FTHG"].to_numpy()
        ag_actual = matches["FTAG"].to_numpy()
        # Use xG as the attack/defense fitting target where we have it,
        # falling back to the actual score for any match that lacks it.
        hg_fit = matches["HxG"].fillna(matches["FTHG"]).to_numpy()
        ag_fit = matches["AxG"].fillna(matches["FTAG"]).to_numpy()

        days_ago = (as_of - matches["Date"]).dt.days.to_numpy().astype(float)
        days_ago = np.clip(days_ago, 0, None)
        weights = np.exp(-xi * days_ago)

        n = self.n
        x0 = np.zeros(2 * (n - 1) + 2)
        x0[2 * (n - 1)] = 0.25   # home_adv start
        x0[2 * (n - 1) + 1] = 0.0  # rho start

        bounds = [(-3, 3)] * (2 * (n - 1)) + [(-2, 2), (-0.4, 0.4)]

        res = minimize(
            self._neg_log_likelihood,
            x0,
            args=(home_idx, away_idx, hg_fit, ag_fit, hg_actual, ag_actual, weights),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if not res.success:
            raise RuntimeError(f"Optimization failed: {res.message}")

        att, deff, home_adv, rho = self._unpack(res.x)
        self.params_ = {
            "att": dict(zip(self.teams, att)),
            "def": dict(zip(self.teams, deff)),
            "home_adv": float(home_adv),
            "rho": float(rho),
        }
        return self

    def rating(self, team):
        return self.params_["att"][team], self.params_["def"][team]

    def score_matrix(self, home_team, away_team, max_goals=MAX_GOALS):
        att_h, def_h = self.rating(home_team)
        att_a, def_a = self.rating(away_team)
        home_adv = self.params_["home_adv"]
        rho = self.params_["rho"]

        lam = np.exp(home_adv + att_h + def_a)
        mu = np.exp(att_a + def_h)

        goals = np.arange(0, max_goals + 1)
        factorials = np.array([math.factorial(g) for g in goals])
        p_home = np.exp(-lam) * lam ** goals / factorials
        p_away = np.exp(-mu) * mu ** goals / factorials
        matrix = np.outer(p_home, p_away)

        for x in (0, 1):
            for y in (0, 1):
                matrix[x, y] *= self._tau(x, y, lam, mu, rho)

        matrix = np.clip(matrix, 0, None)
        matrix /= matrix.sum()
        return matrix, lam, mu


# ---------------------------------------------------------------------------
# 4. Promoted-team baseline ratings (18th-place historical average)
# ---------------------------------------------------------------------------

def eighteenth_place_baseline(training_data: pd.DataFrame, model: DixonColes):
    """Average the fitted (att, def) of the team that finished 18th in each
    training season, to use as a baseline for teams with no recent top-flight
    data (freshly promoted clubs)."""
    atts, defs = [], []
    for code in TRAIN_SEASON_CODES:
        season_df = training_data[training_data["SeasonCode"] == code]
        table = season_table(season_df)
        eighteenth = table.iloc[17]["Team"]  # 0-indexed: 18th place
        a, d = model.rating(eighteenth)
        atts.append(a)
        defs.append(d)
    return float(np.mean(atts)), float(np.mean(defs))


# ---------------------------------------------------------------------------
# 5. Real fixture list (official 2026-27 schedule, snapshotted locally)
# ---------------------------------------------------------------------------

FIXTURES_FILE = "fixtures_2026_27_raw.json"

# Maps the fixture-list source's team slugs to football-data.co.uk's team
# names (the names our fitted model's ratings are keyed on).
SLUG_TO_TEAM = {
    "afc-bournemouth": "Bournemouth",
    "arsenal": "Arsenal",
    "aston-villa": "Aston Villa",
    "brentford": "Brentford",
    "brighton-and-hove-albion": "Brighton",
    "chelsea": "Chelsea",
    "coventry-city": "Coventry",
    "crystal-palace": "Crystal Palace",
    "everton": "Everton",
    "fulham": "Fulham",
    "hull-city": "Hull",
    "ipswich-town": "Ipswich",
    "leeds-united": "Leeds",
    "liverpool": "Liverpool",
    "manchester-city": "Man City",
    "manchester-united": "Man United",
    "newcastle-united": "Newcastle",
    "nottingham-forest": "Nott'm Forest",
    "sunderland": "Sunderland",
    "tottenham-hotspur": "Tottenham",
}


def load_real_fixtures(path: str = FIXTURES_FILE) -> pd.DataFrame:
    """Load the official 2026-27 fixture list (380 matches, real matchweeks
    and dates), snapshotted to a local JSON file so the pipeline stays
    self-contained and doesn't depend on scraping a live page each run.

    The file was built once from the official schedule and cross-checked
    against already-played 2026-27 results from football-data.co.uk (exact
    match for gameweeks 1-2) -- see README for provenance.
    """
    with open(path) as f:
        raw = pd.DataFrame(pd.read_json(f))
    raw["HomeTeam"] = raw["home"].map(SLUG_TO_TEAM)
    raw["AwayTeam"] = raw["away"].map(SLUG_TO_TEAM)
    if raw["HomeTeam"].isna().any() or raw["AwayTeam"].isna().any():
        unmapped = set(raw.loc[raw["HomeTeam"].isna(), "home"]) | set(
            raw.loc[raw["AwayTeam"].isna(), "away"]
        )
        raise ValueError(f"Unmapped team slug(s) in fixture file: {unmapped}")

    fixtures = raw.rename(columns={"week": "Gameweek", "date": "Date"})
    fixtures = fixtures[["Gameweek", "Date", "HomeTeam", "AwayTeam"]]
    fixtures = fixtures.sort_values(["Gameweek", "Date"]).reset_index(drop=True)
    return fixtures


# ---------------------------------------------------------------------------
# 6. Reusable pipeline entry point (fit model + load real fixtures)
# ---------------------------------------------------------------------------

def build_model_and_fixtures(verbose: bool = True):
    """Download data, fit the Dixon-Coles model (with promoted-team baseline
    ratings applied), and load the real 380-fixture schedule. Shared by
    model.py (single-match predictions) and simulate_season.py (full-season
    Monte Carlo projection) so both stay consistent."""
    def log(msg):
        if verbose:
            print(msg)

    log("Downloading historical Premier League data (2023-24 to 2025-26)...")
    training_data = load_training_data()
    log(f"  {len(training_data)} matches loaded.")

    log("Downloading current-season data to determine 2026-27 team list...")
    current_teams = load_current_season_teams()
    log(f"  {len(current_teams)} teams: {current_teams}")

    trained_teams = sorted(set(training_data["HomeTeam"]) | set(training_data["AwayTeam"]))
    new_teams = sorted(set(current_teams) - set(trained_teams))
    if new_teams:
        log(f"  Teams with no recent PL history (baseline rating applied): {new_teams}")

    log("Fitting Dixon-Coles model (MLE, exponential time decay)...")
    model = DixonColes(trained_teams)
    as_of = training_data["Date"].max() + pd.Timedelta(days=1)
    model.fit(training_data, as_of=as_of, xi=XI)
    log(f"  home_adv={model.params_['home_adv']:.3f}  rho={model.params_['rho']:.3f}")

    if new_teams:
        base_att, base_def = eighteenth_place_baseline(training_data, model)
        log(f"  18th-place baseline rating: att={base_att:.3f}, def={base_def:.3f}")
        for t in new_teams:
            model.params_["att"][t] = base_att
            model.params_["def"][t] = base_def
            model.teams.append(t)

    log("Loading real 2026-27 fixture list (380 matches)...")
    fixtures = load_real_fixtures()
    if set(fixtures["HomeTeam"]) | set(fixtures["AwayTeam"]) != set(current_teams):
        raise ValueError("Fixture-file team list doesn't match current-season team list.")

    return model, fixtures, new_teams


def main():
    model, fixtures, _ = build_model_and_fixtures()

    print("Simulating all fixtures...")
    rows = []
    for _, f in fixtures.iterrows():
        home, away = f["HomeTeam"], f["AwayTeam"]
        matrix, lam, mu = model.score_matrix(home, away)

        flat_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
        home_goals, away_goals = int(flat_idx[0]), int(flat_idx[1])
        most_likely_score = f"{home_goals}-{away_goals}"

        p_home = np.tril(matrix, -1).sum()
        p_draw = np.trace(matrix)
        p_away = np.triu(matrix, 1).sum()

        if p_home >= p_draw and p_home >= p_away:
            favorite_pct, blurb = p_home, f"{p_home:.0%} {home} win"
        elif p_away >= p_draw:
            favorite_pct, blurb = p_away, f"{p_away:.0%} {away} win"
        else:
            favorite_pct, blurb = p_draw, f"{p_draw:.0%} draw"
        summary = f"{home} {home_goals}-{away_goals} {away} ({blurb})"

        rows.append({
            "Gameweek": int(f["Gameweek"]),
            "Date": f["Date"].date().isoformat() if pd.notna(f["Date"]) else "",
            "HomeTeam": home,
            "AwayTeam": away,
            "Expected_Home_Goals": round(float(lam), 3),
            "Expected_Away_Goals": round(float(mu), 3),
            "Most_Likely_Score": most_likely_score,
            "P_Home": round(float(p_home), 4),
            "P_Draw": round(float(p_draw), 4),
            "P_Away": round(float(p_away), 4),
            "Summary": summary,
        })

    output = pd.DataFrame(rows)
    out_path = "predictions_2026_2027.csv"
    output.to_csv(out_path, index=False)
    print(f"Wrote {len(output)} predictions to {out_path}")


if __name__ == "__main__":
    main()
