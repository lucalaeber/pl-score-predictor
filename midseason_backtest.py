"""
Mid-season backtest: replay a past season at the point the live model is at
right now (after `origin` matches, ~9 gameweeks), train on the prior seasons
plus those matches, then score every remaining match of that season.

It answers two questions the pre-season backtest can't:
  1. Should a promoted club's own early results be blended with the generic
     18th-place baseline (instead of the baseline replacing them outright)?
  2. Do in-season matches deserve more weight than the mild time decay gives?

Variants compared (all fit on actual goals so this runs from
football-data.co.uk alone; rerun with Understat reachable for xG variants):
  pre-season  prior seasons only, promoted clubs on the baseline
  production  + in-season matches, but only those between known clubs
              (what model.py does today)
  mult=m      production with in-season matches weighted m times
  shrink=s    all in-season matches included; promoted clubs' ratings are
              pulled toward the baseline with a prior of sd s
  both        shrinkage and weighting together

Run:
    python midseason_backtest.py [origin_matches=90] [n_train=5]
"""

import sys

import numpy as np
import pandas as pd

from model import DixonColes, XI, load_season, eighteenth_place_baseline
from backtest import season_codes_before, predict_matches, compute_match_metrics

TARGETS = ["2122", "2223", "2324", "2425", "2526"]
SHRINK_SDS = [0.15, 0.3, 0.6]
MULTS = [2, 3, 5]


def fit_model(train_df, teams, as_of, **kwargs):
    model = DixonColes(sorted(teams))
    model.fit(train_df, as_of=as_of, xi=XI, use_xg=False, **kwargs)
    return model


def apply_baseline(model, new_teams, base_att, base_def):
    for t in new_teams:
        model.params_["att"][t] = base_att
        model.params_["def"][t] = base_def
        model.teams.append(t)


def run_target(target_code, n_train, origin):
    prior_df = pd.concat(
        [load_season(c) for c in season_codes_before(target_code, n_train)], ignore_index=True
    ).sort_values("Date").reset_index(drop=True)
    target = load_season(target_code).sort_values("Date").reset_index(drop=True)
    inseason, test = target.iloc[:origin].copy(), target.iloc[origin:].copy()

    historical = set(prior_df["HomeTeam"]) | set(prior_df["AwayTeam"])
    new_teams = sorted((set(target["HomeTeam"]) | set(target["AwayTeam"])) - historical)
    known_only = inseason[inseason["HomeTeam"].isin(historical) & inseason["AwayTeam"].isin(historical)]

    prior_asof = prior_df["Date"].max() + pd.Timedelta(days=1)
    season_asof = inseason["Date"].max() + pd.Timedelta(days=1)
    outputs = {}

    # pre-season only
    m = fit_model(prior_df, historical, prior_asof)
    base_att, base_def = eighteenth_place_baseline(prior_df, m)
    apply_baseline(m, new_teams, base_att, base_def)
    outputs["pre-season"] = predict_matches(m, test)

    # production, optionally with in-season weighting
    train_prod = pd.concat([prior_df, known_only], ignore_index=True)
    for mult in [1] + MULTS:
        extra = np.where(np.arange(len(train_prod)) >= len(prior_df), mult, 1.0)
        m = fit_model(train_prod, historical, season_asof, extra_weight=extra)
        apply_baseline(m, new_teams, base_att, base_def)
        outputs["production" if mult == 1 else f"mult={mult}"] = predict_matches(m, test)

    # shrinkage toward the baseline, optionally with in-season weighting
    train_all = pd.concat([prior_df, inseason], ignore_index=True)
    prior = {t: (base_att, base_def) for t in new_teams}
    for sd in SHRINK_SDS:
        for mult in [1] + [3]:
            extra = np.where(np.arange(len(train_all)) >= len(prior_df), mult, 1.0)
            m = fit_model(train_all, historical | set(new_teams), season_asof,
                          extra_weight=extra, prior=prior, prior_strength=1 / (2 * sd ** 2))
            name = f"shrink={sd}" if mult == 1 else f"both(s={sd},m={mult})"
            outputs[name] = predict_matches(m, test)

    return outputs, prior_df, new_teams


def score(results, prior_df, new_teams):
    m = compute_match_metrics(results, prior_df)
    probs = results[["P_Home", "P_Draw", "P_Away"]].to_numpy().clip(1e-10, 1)
    idx = results["ActualResult"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    per_match_ll = -np.log(probs[np.arange(len(results)), idx])
    is_new = results["HomeTeam"].isin(new_teams) | results["AwayTeam"].isin(new_teams)
    return {
        "log_loss": m["log_loss"], "brier": m["brier"], "result_acc": m["result_acc"],
        "ll_sum_new": per_match_ll[is_new.to_numpy()].sum(), "n_new": int(is_new.sum()),
    }


def main():
    origin = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    print(f"Mid-season replay: train on {n_train} prior seasons + first {origin} matches, "
          f"score the rest. Targets: {TARGETS}\n")

    per_variant = {}
    for target in TARGETS:
        outputs, prior_df, new_teams = run_target(target, n_train, origin)
        print(f"{target}: new clubs {new_teams}, {len(next(iter(outputs.values())))} test matches")
        for name, res in outputs.items():
            per_variant.setdefault(name, []).append(score(res, prior_df, new_teams))

    print(f"\n{'variant':<22}{'logloss':>9}{'brier':>8}{'result':>8}{'new-club logloss':>19}")
    print("-" * 66)
    for name, rows in per_variant.items():
        ll = np.mean([r["log_loss"] for r in rows])
        br = np.mean([r["brier"] for r in rows])
        acc = np.mean([r["result_acc"] for r in rows])
        new_ll = sum(r["ll_sum_new"] for r in rows) / sum(r["n_new"] for r in rows)
        print(f"{name:<22}{ll:>9.4f}{br:>8.4f}{acc:>7.1%}{new_ll:>13.4f} ({sum(r['n_new'] for r in rows)} matches)")


if __name__ == "__main__":
    main()
