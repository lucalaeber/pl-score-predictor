"""
Backtest: train the model on N seasons ending before a target season, predict
every match of that target season using only pre-season information, then
compare the predictions against what actually happened.

This never lets the model see the target season's own results while fitting
or while computing "naive baseline" comparison rates -- those come only from
the training seasons, exactly like a real bettor/analyst would have to do it
before a ball is kicked.

Run:
    python backtest.py [target_season_code] [n_train_seasons]

Example:
    python backtest.py 2526 3
    -> trains on 2223, 2324, 2425; evaluates against the real 2025-26 season
       (the most recent completed season, so this mirrors exactly what the
       production model does for 2026-27 one season later)

Output:
    backtest_<target>.csv       -- every match: predicted vs actual
    backtest_<target>_table.csv -- projected vs actual final table
"""

import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from model import DixonColes, XI, load_season, attach_understat_xg, eighteenth_place_baseline, season_table
from simulate_season import simulate_season


def season_codes_before(target_code: str, n: int) -> list:
    """e.g. season_codes_before('2526', 3) -> ['2223', '2324', '2425']"""
    start_year = int(target_code[:2])
    return [f"{start_year - i:02d}{(start_year - i + 1) % 100:02d}" for i in range(n, 0, -1)]


def fit_backtest_model(train_codes, use_xg=True, xi=XI, l2=0.0):
    frames = []
    for code in train_codes:
        df = load_season(code)
        df = attach_understat_xg(df, code)
        frames.append(df)
    historical_teams = set()
    for f in frames:
        historical_teams |= set(f["HomeTeam"]) | set(f["AwayTeam"])
    training_data = pd.concat(frames, ignore_index=True).sort_values("Date").reset_index(drop=True)

    model = DixonColes(sorted(historical_teams))
    as_of = training_data["Date"].max() + pd.Timedelta(days=1)
    model.fit(training_data, as_of=as_of, xi=xi, use_xg=use_xg, l2=l2)
    return model, training_data, historical_teams


def predict_matches(model, test_df):
    rows = []
    for _, r in test_df.iterrows():
        home, away = r["HomeTeam"], r["AwayTeam"]
        matrix, lam, mu = model.score_matrix(home, away)

        flat_idx = np.unravel_index(np.argmax(matrix), matrix.shape)
        pred_score = f"{flat_idx[0]}-{flat_idx[1]}"

        p_home = np.tril(matrix, -1).sum()
        p_draw = np.trace(matrix)
        p_away = np.triu(matrix, 1).sum()

        actual_h, actual_a = int(r["FTHG"]), int(r["FTAG"])
        actual_result = "H" if actual_h > actual_a else ("A" if actual_h < actual_a else "D")
        pred_result = max([("H", p_home), ("D", p_draw), ("A", p_away)], key=lambda t: t[1])[0]

        rows.append({
            "Date": r["Date"].date().isoformat(),
            "HomeTeam": home,
            "AwayTeam": away,
            "ActualScore": f"{actual_h}-{actual_a}",
            "PredictedScore": pred_score,
            "ActualResult": actual_result,
            "PredictedResult": pred_result,
            "P_Home": round(float(p_home), 4),
            "P_Draw": round(float(p_draw), 4),
            "P_Away": round(float(p_away), 4),
            "ExpectedHomeGoals": round(float(lam), 3),
            "ExpectedAwayGoals": round(float(mu), 3),
        })
    return pd.DataFrame(rows)


def compute_match_metrics(results: pd.DataFrame, training_data: pd.DataFrame) -> dict:
    n = len(results)
    exact_acc = (results["PredictedScore"] == results["ActualScore"]).mean()
    result_acc = (results["PredictedResult"] == results["ActualResult"]).mean()

    outcome_idx = results["ActualResult"].map({"H": 0, "D": 1, "A": 2}).to_numpy()
    probs = np.clip(results[["P_Home", "P_Draw", "P_Away"]].to_numpy(), 1e-10, 1)
    log_loss = -np.log(probs[np.arange(n), outcome_idx]).mean()
    onehot = np.zeros_like(probs)
    onehot[np.arange(n), outcome_idx] = 1
    brier = ((probs - onehot) ** 2).sum(axis=1).mean()

    # Naive baselines computed ONLY from training-season base rates -- what
    # you could know before the target season starts, for fair comparison.
    train_h = (training_data["FTHG"] > training_data["FTAG"]).mean()
    train_d = (training_data["FTHG"] == training_data["FTAG"]).mean()
    train_a = (training_data["FTHG"] < training_data["FTAG"]).mean()
    naive_probs = np.tile([train_h, train_d, train_a], (n, 1))
    naive_probs = np.clip(naive_probs, 1e-10, 1)
    naive_log_loss = -np.log(naive_probs[np.arange(n), outcome_idx]).mean()
    naive_brier = ((naive_probs - onehot) ** 2).sum(axis=1).mean()
    naive_result_acc = max(train_h, train_d, train_a)  # always guess the most common result

    most_common_score = training_data.groupby(["FTHG", "FTAG"]).size().idxmax()
    naive_exact_acc = (results["ActualScore"] == f"{most_common_score[0]}-{most_common_score[1]}").mean()

    return {
        "n": n,
        "exact_acc": exact_acc, "naive_exact_acc": naive_exact_acc, "naive_score": most_common_score,
        "result_acc": result_acc, "naive_result_acc": naive_result_acc,
        "log_loss": log_loss, "naive_log_loss": naive_log_loss,
        "brier": brier, "naive_brier": naive_brier,
    }


def report_match_metrics(results: pd.DataFrame, training_data: pd.DataFrame):
    m = compute_match_metrics(results, training_data)
    print(f"Matches evaluated: {m['n']}\n")
    print(f"{'Metric':<28}{'Model':>10}{'Naive baseline':>18}")
    print(f"{'-'*56}")
    print(f"{'Exact scoreline accuracy':<28}{m['exact_acc']:>9.1%} {m['naive_exact_acc']:>17.1%}  (naive = always guess {m['naive_score'][0]}-{m['naive_score'][1]})")
    print(f"{'Match result accuracy':<28}{m['result_acc']:>9.1%} {m['naive_result_acc']:>17.1%}  (naive = always guess the most common result)")
    print(f"{'Log loss (lower better)':<28}{m['log_loss']:>10.4f}{m['naive_log_loss']:>19.4f}")
    print(f"{'Brier score (lower better)':<28}{m['brier']:>10.4f}{m['naive_brier']:>19.4f}")


def report_table_metrics(model, test_df, target_code):
    fixtures = test_df[["HomeTeam", "AwayTeam"]]
    projected = simulate_season(model, fixtures, n_sims=20000)
    actual = season_table(test_df)
    actual["Actual_Rank"] = np.arange(1, len(actual) + 1)

    merged = projected.merge(actual[["Team", "Pts", "Actual_Rank"]], on="Team")
    merged = merged.rename(columns={"Pts": "Actual_Points"})
    merged["Rank_Error"] = (merged["Projected_Rank"] - merged["Actual_Rank"]).abs()
    merged["Points_Error"] = (merged["Expected_Points"] - merged["Actual_Points"]).abs()

    rho, _ = spearmanr(merged["Projected_Rank"], merged["Actual_Rank"])
    mean_rank_error = merged["Rank_Error"].mean()
    mean_points_error = merged["Points_Error"].mean()

    actual_top4 = set(actual.nsmallest(4, "Actual_Rank")["Team"])
    projected_top4 = set(projected.nsmallest(4, "Projected_Rank")["Team"])
    actual_bottom3 = set(actual.nlargest(3, "Actual_Rank")["Team"])
    projected_bottom3 = set(projected.nlargest(3, "Projected_Rank")["Team"])

    print(f"\nFinal table: projected (pre-season) vs actual")
    print(f"{'-'*56}")
    print(f"Spearman rank correlation:   {rho:.3f}  (1.0 = perfect order)")
    print(f"Mean |rank error|:           {mean_rank_error:.2f} places")
    print(f"Mean |points error|:         {mean_points_error:.1f} points")
    print(f"Top-4 correctly identified:  {len(actual_top4 & projected_top4)}/4  {sorted(actual_top4 & projected_top4)}")
    print(f"Bottom-3 correctly identified: {len(actual_bottom3 & projected_bottom3)}/3  {sorted(actual_bottom3 & projected_bottom3)}")

    out = merged[["Team", "Projected_Rank", "Actual_Rank", "Expected_Points", "Actual_Points"]]
    out = out.sort_values("Actual_Rank").reset_index(drop=True)
    out_path = f"backtest_{target_code}_table.csv"
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


def run_backtest(target_code, n_train, use_xg=True, xi=XI, l2=0.0, verbose=True):
    train_codes = season_codes_before(target_code, n_train)
    if verbose:
        print(f"Training on seasons: {train_codes}  (use_xg={use_xg}, xi={xi}, l2={l2})")
        print(f"Testing on season:   {target_code}  (real, completed -- ground truth known)\n")

    model, training_data, historical_teams = fit_backtest_model(train_codes, use_xg=use_xg, xi=xi, l2=l2)
    if verbose:
        print(f"home_adv={model.params_['home_adv']:.3f}  rho={model.params_['rho']:.3f}")

    test_df = load_season(target_code)
    test_teams = sorted(set(test_df["HomeTeam"]) | set(test_df["AwayTeam"]))
    new_teams = sorted(set(test_teams) - historical_teams)
    if new_teams:
        if verbose:
            print(f"Teams with no prior history (baseline applied): {new_teams}")
        base_att, base_def = eighteenth_place_baseline(training_data, model)
        for t in new_teams:
            model.params_["att"][t] = base_att
            model.params_["def"][t] = base_def
            model.teams.append(t)

    results = predict_matches(model, test_df)
    return model, results, training_data, test_df


def compare_xg_vs_goals(target_code, n_train, xi=XI):
    print(f"Comparing xG-fit vs goals-fit on season {target_code} ({n_train}-season training window, xi={xi})\n")
    print("=" * 30, "FIT ON ACTUAL GOALS", "=" * 30)
    _, results_g, training_g, _ = run_backtest(target_code, n_train, use_xg=False, xi=xi)
    m_goals = compute_match_metrics(results_g, training_g)

    print()
    print("=" * 30, "FIT ON UNDERSTAT xG", "=" * 30)
    _, results_x, training_x, _ = run_backtest(target_code, n_train, use_xg=True, xi=xi)
    m_xg = compute_match_metrics(results_x, training_x)

    print(f"\n{'='*70}")
    print(f"{'Metric':<28}{'Goals-fit':>14}{'xG-fit':>14}{'Better':>14}")
    print("-" * 70)
    rows = [
        ("Exact scoreline accuracy", m_goals["exact_acc"], m_xg["exact_acc"], "higher"),
        ("Match result accuracy", m_goals["result_acc"], m_xg["result_acc"], "higher"),
        ("Log loss", m_goals["log_loss"], m_xg["log_loss"], "lower"),
        ("Brier score", m_goals["brier"], m_xg["brier"], "lower"),
    ]
    for label, g, x, direction in rows:
        better = "xG" if (x > g) == (direction == "higher") else "Goals"
        if x == g:
            better = "tie"
        fmt = "{:>13.1%}" if "accuracy" in label else "{:>13.4f} "
        print(f"{label:<28}{fmt.format(g):>14}{fmt.format(x):>14}{better:>14}")


def sweep_xi(target_code, n_train, use_xg=True):
    xi_values = [0.0, 0.001, 0.002, 0.003, 0.0065, 0.01, 0.02, 0.04]
    print(f"Sweeping xi (time-decay rate) on season {target_code}, use_xg={use_xg}\n")
    print(f"{'xi':>8}{'home_adv':>10}{'rho':>8}{'ResultAcc':>12}{'LogLoss':>10}{'Brier':>9}")
    print("-" * 57)
    best = None
    for xi in xi_values:
        model, results, training_data, _ = run_backtest(target_code, n_train, use_xg=use_xg, xi=xi, verbose=False)
        m = compute_match_metrics(results, training_data)
        print(f"{xi:>8.4f}{model.params_['home_adv']:>10.3f}{model.params_['rho']:>8.3f}"
              f"{m['result_acc']:>11.1%} {m['log_loss']:>9.4f}{m['brier']:>9.4f}")
        if best is None or m["log_loss"] < best[1]:
            best = (xi, m["log_loss"])
    print(f"\nLowest log loss at xi={best[0]} ({best[1]:.4f})")


def sweep_l2(target_code, n_train, use_xg=True, xi=XI):
    l2_values = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
    print(f"Sweeping l2 (attack/defense shrinkage) on season {target_code}, use_xg={use_xg}, xi={xi}\n")
    print(f"{'l2':>8}{'home_adv':>10}{'rho':>8}{'ResultAcc':>12}{'LogLoss':>10}{'Brier':>9}")
    print("-" * 57)
    best = None
    for l2 in l2_values:
        model, results, training_data, _ = run_backtest(target_code, n_train, use_xg=use_xg, xi=xi, l2=l2, verbose=False)
        m = compute_match_metrics(results, training_data)
        print(f"{l2:>8.3f}{model.params_['home_adv']:>10.3f}{model.params_['rho']:>8.3f}"
              f"{m['result_acc']:>11.1%} {m['log_loss']:>9.4f}{m['brier']:>9.4f}")
        if best is None or m["log_loss"] < best[1]:
            best = (l2, m["log_loss"])
    print(f"\nLowest log loss at l2={best[0]} ({best[1]:.4f})")


def main():
    target_code = sys.argv[1] if len(sys.argv) > 1 else "2526"
    n_train = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    mode = sys.argv[3] if len(sys.argv) > 3 else "run"

    if mode == "compare":
        xi = float(sys.argv[4]) if len(sys.argv) > 4 else XI
        compare_xg_vs_goals(target_code, n_train, xi=xi)
        return
    if mode == "sweep-xi":
        sweep_xi(target_code, n_train)
        return
    if mode == "sweep-l2":
        xi = float(sys.argv[4]) if len(sys.argv) > 4 else XI
        sweep_l2(target_code, n_train, xi=xi)
        return

    model, results, training_data, test_df = run_backtest(target_code, n_train, use_xg=True)
    out_path = f"backtest_{target_code}.csv"
    results.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}\n")

    report_match_metrics(results, training_data)
    report_table_metrics(model, test_df, target_code)


if __name__ == "__main__":
    main()
