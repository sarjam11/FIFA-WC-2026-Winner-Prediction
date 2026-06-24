
import pandas as pd
import numpy as np
from datetime import timedelta
import os
import warnings

warnings.filterwarnings("ignore")

OUTPUT_DIR = "C:/Users/Sarjam/OneDrive/Desktop/FIFA WC Prediction/feature engineering"


# CONFIGURATION


TEAM_NAME_MAP = {
    "USA": "United States",
    "Turkiye": "Turkey",
    "Czechia": "Czech Republic",
    "Bosnia": "Bosnia and Herzegovina",
    "Curacao": "Curaçao",
}

# Elo — NO home advantage (all neutral)
ELO_INITIAL  = 1500
ELO_K_FACTOR = 40
ELO_HOME_ADV = 0  # neutral venue — no advantage

MATCH_WEIGHTS = {
    "FIFA World Cup": 8.0,
    "FIFA World Cup qualification": 4.0,
    "UEFA Euro": 6.0,
    "UEFA Euro qualification": 3.0,
    "Copa América": 6.0,
    "African Cup of Nations": 5.0,
    "African Cup of Nations qualification": 3.0,
    "AFC Asian Cup": 5.0,
    "AFC Asian Cup qualification": 3.0,
    "CONCACAF Nations League": 3.5,
    "Gold Cup": 4.0,
    "UEFA Nations League": 3.5,
    "Friendly": 1.0,
    "FIFA Series": 1.5,
    "CONCACAF Series": 1.5,
}
DEFAULT_WEIGHT = 2.0

ROLL_SHORT = 10
ROLL_LONG  = 20



# LOAD & CLEAN


def load_and_clean():
    print("=" * 70)
    print("  PHASE 2: FEATURE ENGINEERING")
    print("=" * 70)
    print("\n[1/9] Loading & cleaning data...")

    matches = pd.read_csv("C:/Users/Sarjam/OneDrive/Desktop/FIFA WC Prediction/filtered_results.csv")
    matches["date"] = pd.to_datetime(matches["date"], dayfirst=True)

    played = matches[matches["team_1_score"].notna()].copy()
    future = matches[matches["team_1_score"].isna()].copy()

    played["team_1_score"] = played["team_1_score"].astype(int)
    played["team_2_score"] = played["team_2_score"].astype(int)

    played["result"] = np.where(
        played["team_1_score"] > played["team_2_score"], "W1",
        np.where(played["team_1_score"] < played["team_2_score"], "W2", "D")
    )

    played["match_weight"] = played["tournament"].map(MATCH_WEIGHTS).fillna(DEFAULT_WEIGHT)
    played["is_competitive"] = ~played["tournament"].str.contains(
        "Friendly|FIFA Series|CONCACAF Series", case=False, na=False
    )

    print(f"  Played matches:  {len(played):,}")
    print(f"  Future fixtures: {len(future):,} (WC 2026 group stage)")

    players = pd.read_csv("C:/Users/Sarjam/OneDrive/Desktop/FIFA WC Prediction/merged_player_stats.csv")
    players["Team_Name"] = players["Team_Name"].replace(TEAM_NAME_MAP)
    players = players.dropna(subset=["Team_Name"])

    print(f"  Players loaded:  {len(players):,} across {players['Team_Name'].nunique()} teams")
    return played, future, players



# ELO RATINGS (no home advantage)


def compute_elo(played: pd.DataFrame):
    print("\n[2/9] Computing Elo ratings (neutral venue, no home advantage)...")

    elo = {}
    records = []

    for _, row in played.sort_values("date").iterrows():
        t1, t2 = row["team_1"], row["team_2"]
        elo.setdefault(t1, ELO_INITIAL)
        elo.setdefault(t2, ELO_INITIAL)

        e1, e2 = elo[t1], elo[t2]

        # No home advantage — symmetric expected scores
        exp1 = 1 / (1 + 10 ** ((e2 - e1) / 400))
        exp2 = 1 - exp1

        actual = {"W1": (1, 0), "D": (0.5, 0.5), "W2": (0, 1)}[row["result"]]

        k = ELO_K_FACTOR * (row["match_weight"] / 4.0)
        gd = abs(row["team_1_score"] - row["team_2_score"])
        gd_mult = np.log(max(gd, 1) + 1)

        d1 = k * gd_mult * (actual[0] - exp1)
        d2 = k * gd_mult * (actual[1] - exp2)

        records.append({"date": row["date"], "team": t1, "elo_before": e1, "elo_after": e1 + d1})
        records.append({"date": row["date"], "team": t2, "elo_before": e2, "elo_after": e2 + d2})

        elo[t1] += d1
        elo[t2] += d2

    elo_df = pd.DataFrame(records)

    top = sorted(elo.items(), key=lambda x: -x[1])[:15]
    print(f"  Elo computed for {len(elo)} teams")
    print(f"  Top 15 entering WC 2026:")
    for i, (team, rating) in enumerate(top, 1):
        print(f"    {i:2d}. {team:<25s} {rating:.0f}")

    return elo, elo_df


def get_elo_at_date(elo_df, team, date):
    mask = (elo_df["team"] == team) & (elo_df["date"] < date)
    sub = elo_df.loc[mask]
    return sub.iloc[-1]["elo_after"] if len(sub) > 0 else ELO_INITIAL



# TEAM MATCH LOG


def build_team_log(played):
    print("\n[3/9] Building team match log...")

    side1 = played.copy()
    side1["team"]        = side1["team_1"]
    side1["opponent"]    = side1["team_2"]
    side1["gf"]          = side1["team_1_score"]
    side1["ga"]          = side1["team_2_score"]
    side1["result_team"] = side1["result"].map({"W1": "W", "W2": "L", "D": "D"})

    side2 = played.copy()
    side2["team"]        = side2["team_2"]
    side2["opponent"]    = side2["team_1"]
    side2["gf"]          = side2["team_2_score"]
    side2["ga"]          = side2["team_1_score"]
    side2["result_team"] = side2["result"].map({"W1": "L", "W2": "W", "D": "D"})

    fields = ["date", "team", "opponent", "gf", "ga",
              "result_team", "tournament", "match_weight", "is_competitive"]

    log = pd.concat([side1[fields], side2[fields]], ignore_index=True)
    log = log.sort_values(["team", "date"]).reset_index(drop=True)
    print(f"  {len(log):,} team-match rows ({log['team'].nunique()} teams)")
    return log



# ROLLING FEATURES


def compute_rolling(team_log):
    print("\n[4/9] Computing rolling features...")

    all_feats = []

    for team, grp in team_log.groupby("team"):
        grp = grp.sort_values("date").reset_index(drop=True)
        n = len(grp)

        gf   = grp["gf"].values
        ga   = grp["ga"].values
        res  = grp["result_team"].values
        comp = grp["is_competitive"].values

        feat = {
            "date": grp["date"].values,
            "team": [team] * n,
            f"avg_gf_{ROLL_SHORT}":        np.full(n, np.nan),
            f"avg_ga_{ROLL_SHORT}":        np.full(n, np.nan),
            f"avg_gd_{ROLL_SHORT}":        np.full(n, np.nan),
            f"avg_gf_{ROLL_LONG}":         np.full(n, np.nan),
            f"avg_ga_{ROLL_LONG}":         np.full(n, np.nan),
            f"win_rate_{ROLL_SHORT}":      np.full(n, np.nan),
            f"win_rate_{ROLL_SHORT}_comp": np.full(n, np.nan),
            f"draw_rate_{ROLL_SHORT}":     np.full(n, np.nan),
            "clean_sheet_rate":            np.full(n, np.nan),
            "unbeaten_run":                np.zeros(n, dtype=int),
            "winning_streak":              np.zeros(n, dtype=int),
            "losing_streak":               np.zeros(n, dtype=int),
        }

        for i in range(1, n):
            s = max(0, i - ROLL_SHORT)
            w_gf, w_ga, w_res = gf[s:i], ga[s:i], res[s:i]
            feat[f"avg_gf_{ROLL_SHORT}"][i] = w_gf.mean()
            feat[f"avg_ga_{ROLL_SHORT}"][i] = w_ga.mean()
            feat[f"avg_gd_{ROLL_SHORT}"][i] = (w_gf - w_ga).mean()
            feat[f"win_rate_{ROLL_SHORT}"][i] = (w_res == "W").mean()
            feat[f"draw_rate_{ROLL_SHORT}"][i] = (w_res == "D").mean()
            feat["clean_sheet_rate"][i] = (w_ga == 0).mean()

            l = max(0, i - ROLL_LONG)
            feat[f"avg_gf_{ROLL_LONG}"][i] = gf[l:i].mean()
            feat[f"avg_ga_{ROLL_LONG}"][i] = ga[l:i].mean()

            comp_res = res[:i][comp[:i]]
            if len(comp_res) >= 3:
                recent = comp_res[-ROLL_SHORT:]
                feat[f"win_rate_{ROLL_SHORT}_comp"][i] = (recent == "W").mean()

            for j in range(i - 1, -1, -1):
                if res[j] != "L": feat["unbeaten_run"][i] += 1
                else: break
            for j in range(i - 1, -1, -1):
                if res[j] == "W": feat["winning_streak"][i] += 1
                else: break
            for j in range(i - 1, -1, -1):
                if res[j] == "L": feat["losing_streak"][i] += 1
                else: break

        all_feats.append(pd.DataFrame(feat))

    result = pd.concat(all_feats, ignore_index=True)
    print(f"  Done — {result.shape[1] - 2} rolling features")
    return result


def get_latest_rolling(rolling_df, team, before_date):
    mask = (rolling_df["team"] == team) & (rolling_df["date"] < before_date)
    sub = rolling_df.loc[mask]
    if sub.empty:
        return {}
    row = sub.iloc[-1]
    return {c: row[c] for c in rolling_df.columns if c not in ("date", "team")}



# HEAD-TO-HEAD FEATURES


def compute_h2h(played):
    print("\n[5/9] Computing head-to-head features...")

    played_sorted = played.sort_values("date").reset_index(drop=True)
    h2h_cache = {}
    records = []

    for _, row in played_sorted.iterrows():
        t1, t2, date = row["team_1"], row["team_2"], row["date"]

        past_12 = h2h_cache.get((t1, t2), [])
        past_21 = h2h_cache.get((t2, t1), [])

        # From t1's perspective
        gf_list = [g[1] for g in past_12] + [g[2] for g in past_21]
        ga_list = [g[2] for g in past_12] + [g[1] for g in past_21]
        total = len(gf_list)

        wins  = sum(1 for f, a in zip(gf_list, ga_list) if f > a)
        draws = sum(1 for f, a in zip(gf_list, ga_list) if f == a)

        rec = {
            "date": date, "team_1": t1, "team_2": t2,
            "h2h_total":       total,
            "h2h_t1_win_pct":  wins / total if total else 0.5,
            "h2h_draw_pct":    draws / total if total else 0.25,
            "h2h_avg_gf":      np.mean(gf_list) if gf_list else np.nan,
            "h2h_avg_ga":      np.mean(ga_list) if ga_list else np.nan,
            "h2h_avg_gd":      np.mean([f - a for f, a in zip(gf_list, ga_list)]) if gf_list else 0.0,
        }

        if gf_list:
            r5_gf, r5_ga = gf_list[-5:], ga_list[-5:]
            rec["h2h_recent_win_pct"] = sum(1 for f, a in zip(r5_gf, r5_ga) if f > a) / len(r5_gf)
        else:
            rec["h2h_recent_win_pct"] = 0.5

        records.append(rec)
        h2h_cache.setdefault((t1, t2), []).append((date, row["team_1_score"], row["team_2_score"]))

    h2h_df = pd.DataFrame(records)
    print(f"  Done — {h2h_df.shape[1] - 3} H2H features")
    return h2h_df


def get_h2h_for_fixture(played, t1, t2):
    mask_12 = (played["team_1"] == t1) & (played["team_2"] == t2)
    mask_21 = (played["team_1"] == t2) & (played["team_2"] == t1)

    gf_list, ga_list = [], []
    for _, r in played[mask_12].iterrows():
        gf_list.append(r["team_1_score"]); ga_list.append(r["team_2_score"])
    for _, r in played[mask_21].iterrows():
        gf_list.append(r["team_2_score"]); ga_list.append(r["team_1_score"])

    total = len(gf_list)
    wins  = sum(1 for f, a in zip(gf_list, ga_list) if f > a)
    draws = sum(1 for f, a in zip(gf_list, ga_list) if f == a)

    rec = {
        "h2h_total":       total,
        "h2h_t1_win_pct":  wins / total if total else 0.5,
        "h2h_draw_pct":    draws / total if total else 0.25,
        "h2h_avg_gf":      np.mean(gf_list) if gf_list else np.nan,
        "h2h_avg_ga":      np.mean(ga_list) if ga_list else np.nan,
        "h2h_avg_gd":      np.mean([f - a for f, a in zip(gf_list, ga_list)]) if gf_list else 0.0,
    }
    if gf_list:
        r5_gf, r5_ga = gf_list[-5:], ga_list[-5:]
        rec["h2h_recent_win_pct"] = sum(1 for f, a in zip(r5_gf, r5_ga) if f > a) / len(r5_gf)
    else:
        rec["h2h_recent_win_pct"] = 0.5
    return rec



# SQUAD QUALITY FEATURES


def compute_squad_features(players):
    print("\n[6/9] Engineering squad quality features from player data...")

    agg = players.groupby("Team_Name").agg(
        squad_size            = ("Player_ID", "count"),
        squad_avg_rating      = ("rating", "mean"),
        squad_max_rating      = ("rating", "max"),
        squad_median_rating   = ("rating", "median"),
        squad_rating_std      = ("rating", "std"),
        squad_total_apps      = ("apps", "sum"),
        squad_avg_apps        = ("apps", "mean"),
        squad_max_apps        = ("apps", "max"),
        squad_total_goals     = ("goal", "sum"),
        squad_total_assists   = ("assistTotal", "sum"),
        squad_avg_shots_pg    = ("shotsPerGame", "mean"),
        squad_avg_keypasses   = ("keyPassPerGame", "mean"),
        squad_avg_dribbles    = ("dribbleWonPerGame", "mean"),
        squad_avg_tackles     = ("tacklePerGame", "mean"),
        squad_avg_intercept   = ("interceptionPerGame", "mean"),
        squad_avg_clearances  = ("clearancePerGame", "mean"),
        squad_avg_blocks      = ("outfielderBlockPerGame", "mean"),
        squad_total_yellows   = ("yellowCard", "sum"),
        squad_total_reds      = ("redCard", "sum"),
        squad_avg_fouls_given = ("foulGivenPerGame", "mean"),
        squad_avg_pass_pct    = ("passSuccess", "mean"),
        squad_avg_passes_pg   = ("totalPassesPerGame", "mean"),
        squad_avg_long_passes = ("accurateLongPassPerGame", "mean"),
        squad_avg_through     = ("accurateThroughBallPerGame", "mean"),
        squad_motm_total      = ("manOfTheMatch", "sum"),
    ).reset_index().rename(columns={"Team_Name": "team"})

    agg["squad_gc_per_app"] = (agg["squad_total_goals"] + agg["squad_total_assists"]) / agg["squad_total_apps"]
    agg["squad_off_def_ratio"] = agg["squad_avg_shots_pg"] / (agg["squad_avg_tackles"] + 0.01)
    agg["squad_cards_per_app"] = (agg["squad_total_yellows"] + 3 * agg["squad_total_reds"]) / agg["squad_total_apps"]
    agg["squad_star_concentration"] = agg["squad_max_rating"] / (agg["squad_avg_rating"] + 0.01)

    exp_depth = players.groupby("Team_Name").apply(
        lambda x: (x["apps"] >= 30).mean(), include_groups=False
    ).reset_index()
    exp_depth.columns = ["team", "squad_exp_depth_ratio"]
    agg = agg.merge(exp_depth, on="team", how="left")

    print(f"  {agg.shape[1] - 1} squad features for {len(agg)} teams")
    return agg



# MOMENTUM FEATURES


def compute_momentum(team_log):
    print("\n[7/9] Computing momentum features...")

    records = []

    for team, grp in team_log.groupby("team"):
        grp = grp.sort_values("date").reset_index(drop=True)

        for i in range(len(grp)):
            date = grp.loc[i, "date"]

            q_start = date - timedelta(days=730)
            q_mask = (
                (grp["date"] >= q_start) & (grp["date"] < date) &
                grp["tournament"].str.contains("qualification|qualifying", case=False, na=False)
            )
            qm = grp.loc[q_mask]

            if len(qm) >= 3:
                qw = (qm["result_team"] == "W").sum()
                qd = (qm["result_team"] == "D").sum()
                qual_ppg   = (qw * 3 + qd) / len(qm)
                qual_gf_pg = qm["gf"].mean()
                qual_ga_pg = qm["ga"].mean()
                qual_gd_pg = qual_gf_pg - qual_ga_pg
            else:
                qual_ppg = qual_gf_pg = qual_ga_pg = qual_gd_pg = np.nan

            comp_wins = grp.loc[
                (grp.index < i) & grp["is_competitive"] & (grp["result_team"] == "W")
            ]
            days_since_cw = (date - comp_wins.iloc[-1]["date"]).days if len(comp_wins) else 999

            if i >= 10:
                recent5 = grp.loc[i-5:i-1, "result_team"].values
                prev5   = grp.loc[i-10:i-6, "result_team"].values
                form_delta = (recent5 == "W").mean() - (prev5 == "W").mean()
            elif i >= 5:
                recent5 = grp.loc[i-5:i-1, "result_team"].values
                form_delta = (recent5 == "W").mean() - 0.33
            else:
                form_delta = 0.0

            records.append({
                "date": date, "team": team,
                "qualifying_ppg": qual_ppg, "qualifying_gf_pg": qual_gf_pg,
                "qualifying_ga_pg": qual_ga_pg, "qualifying_gd_pg": qual_gd_pg,
                "days_since_comp_win": days_since_cw, "form_trajectory": form_delta,
            })

    mom_df = pd.DataFrame(records)
    print(f"  Done — {mom_df.shape[1] - 2} momentum features")
    return mom_df



# CONTEXTUAL FEATURES 


def compute_context(df):
    print("\n[8/9] Computing contextual features...")

    ctx = df[["date", "team_1", "team_2", "tournament"]].copy()

    ctx["is_world_cup"]    = ctx["tournament"].str.contains("FIFA World Cup$", regex=True, na=False).astype(int)
    ctx["is_wc_qualifier"] = ctx["tournament"].str.contains("qualification", case=False, na=False).astype(int)
    ctx["is_continental"]  = ctx["tournament"].str.contains(
        "Euro|Copa|African Cup|Asian Cup|Gold Cup|Nations League", case=False, na=False
    ).astype(int)
    ctx["is_friendly"]     = ctx["tournament"].str.contains(
        "Friendly|Series", case=False, na=False
    ).astype(int)

    print(f"  Done — 4 contextual features")
    return ctx



# ASSEMBLE


def assemble_training_features(played, elo_df, rolling_df, h2h_df, momentum_df, context_df, squad_df):
    print("\n[9/9] Assembling training feature matrix...")

    base = played[["date", "team_1", "team_2", "tournament"]].copy().reset_index(drop=True)
    base["_mid"] = base.index

    roll_dedup = rolling_df.drop_duplicates(subset=["date", "team"], keep="last")
    mom_dedup  = momentum_df.drop_duplicates(subset=["date", "team"], keep="last")

    # Elo 
    e1s, e2s = [], []
    for _, r in base.iterrows():
        e1s.append(get_elo_at_date(elo_df, r["team_1"], r["date"]))
        e2s.append(get_elo_at_date(elo_df, r["team_2"], r["date"]))
    base["t1_elo"] = e1s
    base["t2_elo"] = e2s
    base["elo_diff"] = base["t1_elo"] - base["t2_elo"]

    # Rolling t1 
    feat_cols_r = [c for c in roll_dedup.columns if c not in ("date", "team")]
    r1 = roll_dedup.rename(columns={c: f"t1_{c}" for c in feat_cols_r})
    r1 = r1.rename(columns={"team": "team_1"})
    base = base.merge(r1, on=["date", "team_1"], how="left")

    # Rolling t2 
    r2 = roll_dedup.rename(columns={c: f"t2_{c}" for c in feat_cols_r})
    r2 = r2.rename(columns={"team": "team_2"})
    base = base.merge(r2, on=["date", "team_2"], how="left")

    # H2H 
    h2h_cols = [c for c in h2h_df.columns if c.startswith("h2h_")]
    base = base.merge(h2h_df[["date", "team_1", "team_2"] + h2h_cols],
                       on=["date", "team_1", "team_2"], how="left")

    # Momentum t1 
    mcols = [c for c in mom_dedup.columns if c not in ("date", "team")]
    m1 = mom_dedup.rename(columns={c: f"t1_{c}" for c in mcols})
    m1 = m1.rename(columns={"team": "team_1"})
    base = base.merge(m1, on=["date", "team_1"], how="left")

    # Momentum t2
    m2 = mom_dedup.rename(columns={c: f"t2_{c}" for c in mcols})
    m2 = m2.rename(columns={"team": "team_2"})
    base = base.merge(m2, on=["date", "team_2"], how="left")

    # Context 
    ctx_cols = [c for c in context_df.columns if c not in ("date", "team_1", "team_2", "tournament")]
    base = base.merge(context_df[["date", "team_1", "team_2"] + ctx_cols],
                       on=["date", "team_1", "team_2"], how="left")

    # Squad t1 
    sq1 = squad_df.rename(columns={c: f"t1_{c}" for c in squad_df.columns if c != "team"})
    sq1 = sq1.rename(columns={"team": "team_1"})
    base = base.merge(sq1, on="team_1", how="left")

    # Squad t2 
    sq2 = squad_df.rename(columns={c: f"t2_{c}" for c in squad_df.columns if c != "team"})
    sq2 = sq2.rename(columns={"team": "team_2"})
    base = base.merge(sq2, on="team_2", how="left")

    # Deduplicate 
    if len(base) > len(played):
        base = base.drop_duplicates(subset=["_mid"], keep="first")
    base = base.drop(columns=["_mid"]).reset_index(drop=True)

    # Differentials
    diff_feats = [
        f"avg_gf_{ROLL_SHORT}", f"avg_ga_{ROLL_SHORT}", f"avg_gd_{ROLL_SHORT}",
        f"avg_gf_{ROLL_LONG}",
        f"win_rate_{ROLL_SHORT}", f"win_rate_{ROLL_SHORT}_comp",
        "unbeaten_run", "winning_streak",
        "qualifying_ppg",
        "squad_avg_rating", "squad_total_goals",
        "squad_avg_tackles", "squad_avg_pass_pct",
    ]
    for feat in diff_feats:
        c1, c2 = f"t1_{feat}", f"t2_{feat}"
        if c1 in base.columns and c2 in base.columns:
            base[f"diff_{feat}"] = base[c1] - base[c2]

    # Targets 
    played_reset = played.reset_index(drop=True)
    base["target_result"]   = played_reset["result"].map({"W1": 0, "D": 1, "W2": 2}).values
    base["target_t1_goals"] = played_reset["team_1_score"].values
    base["target_t2_goals"] = played_reset["team_2_score"].values
    base["target_gd"]       = base["target_t1_goals"] - base["target_t2_goals"]

    print(f"  Training matrix: {base.shape[0]:,} rows × {base.shape[1]} columns")
    return base


def assemble_fixture_features(future, played, elo_current, elo_df, rolling_df, momentum_df, squad_df):
    print("\n  Building WC 2026 fixture features...")

    rows = []
    cutoff = future["date"].min()

    for _, fix in future.iterrows():
        t1, t2, date = fix["team_1"], fix["team_2"], fix["date"]
        row = {"date": date, "team_1": t1, "team_2": t2}

        e1 = elo_current.get(t1, ELO_INITIAL)
        e2 = elo_current.get(t2, ELO_INITIAL)
        row["t1_elo"] = e1
        row["t2_elo"] = e2
        row["elo_diff"] = e1 - e2

        for side, team in [("t1", t1), ("t2", t2)]:
            latest = get_latest_rolling(rolling_df, team, cutoff)
            for k, v in latest.items():
                row[f"{side}_{k}"] = v

        h2h = get_h2h_for_fixture(played, t1, t2)
        row.update(h2h)

        for side, team in [("t1", t1), ("t2", t2)]:
            mask = (momentum_df["team"] == team) & (momentum_df["date"] < cutoff)
            sub = momentum_df.loc[mask]
            if not sub.empty:
                for c in momentum_df.columns:
                    if c not in ("date", "team"):
                        row[f"{side}_{c}"] = sub.iloc[-1][c]

        row["is_world_cup"]    = 1
        row["is_wc_qualifier"] = 0
        row["is_continental"]  = 0
        row["is_friendly"]     = 0

        for side, team in [("t1", t1), ("t2", t2)]:
            sq = squad_df[squad_df["team"] == team]
            if not sq.empty:
                for c in squad_df.columns:
                    if c != "team":
                        row[f"{side}_{c}"] = sq.iloc[0][c]

        rows.append(row)

    fixture_df = pd.DataFrame(rows)

    diff_feats = [
        f"avg_gf_{ROLL_SHORT}", f"avg_ga_{ROLL_SHORT}", f"avg_gd_{ROLL_SHORT}",
        f"win_rate_{ROLL_SHORT}", f"win_rate_{ROLL_SHORT}_comp",
        "unbeaten_run", "winning_streak",
        "qualifying_ppg", "squad_avg_rating", "squad_total_goals",
        "squad_avg_tackles", "squad_avg_pass_pct",
    ]
    for feat in diff_feats:
        c1, c2 = f"t1_{feat}", f"t2_{feat}"
        if c1 in fixture_df.columns and c2 in fixture_df.columns:
            fixture_df[f"diff_{feat}"] = fixture_df[c1] - fixture_df[c2]

    print(f"  Fixture matrix: {fixture_df.shape[0]} rows × {fixture_df.shape[1]} columns")
    return fixture_df



# TEAM SNAPSHOT


def build_team_snapshot(wc_teams, elo_current, rolling_df, momentum_df, squad_df, cutoff):
    print("\n  Building team snapshot table...")

    rows = []
    for team in sorted(wc_teams):
        row = {"team": team, "elo": elo_current.get(team, ELO_INITIAL)}

        latest = get_latest_rolling(rolling_df, team, cutoff)
        row.update(latest)

        mask = (momentum_df["team"] == team) & (momentum_df["date"] < cutoff)
        sub = momentum_df.loc[mask]
        if not sub.empty:
            for c in momentum_df.columns:
                if c not in ("date", "team"):
                    row[c] = sub.iloc[-1][c]

        sq = squad_df[squad_df["team"] == team]
        if not sq.empty:
            for c in squad_df.columns:
                if c != "team":
                    row[c] = sq.iloc[0][c]

        rows.append(row)

    snap = pd.DataFrame(rows).sort_values("elo", ascending=False).reset_index(drop=True)
    print(f"  Snapshot: {len(snap)} teams × {snap.shape[1]} columns")
    return snap



# MAIN


def main():
    played, future, players = load_and_clean()
    elo_current, elo_df     = compute_elo(played)
    team_log                = build_team_log(played)
    rolling_df              = compute_rolling(team_log)
    h2h_df                  = compute_h2h(played)
    squad_df                = compute_squad_features(players)
    momentum_df             = compute_momentum(team_log)
    context_df              = compute_context(played)

    train_features = assemble_training_features(
        played, elo_df, rolling_df, h2h_df, momentum_df, context_df, squad_df
    )
    fixture_features = assemble_fixture_features(
        future, played, elo_current, elo_df, rolling_df, momentum_df, squad_df
    )

    wc_teams = set(future["team_1"]) | set(future["team_2"])
    cutoff = future["date"].min()
    snapshot = build_team_snapshot(wc_teams, elo_current, rolling_df, momentum_df, squad_df, cutoff)

    # Save 
    train_path   = os.path.join(OUTPUT_DIR, "training_features.csv")
    fixture_path = os.path.join(OUTPUT_DIR, "wc2026_fixture_features.csv")
    snap_path    = os.path.join(OUTPUT_DIR, "team_snapshot.csv")

    train_features.to_csv(train_path, index=False)
    fixture_features.to_csv(fixture_path, index=False)
    snapshot.to_csv(snap_path, index=False)

    print(f"\n  ✓ Saved: {train_path}")
    print(f"  ✓ Saved: {fixture_path}")
    print(f"  ✓ Saved: {snap_path}")

    # Summary
    print("\n" + "=" * 70)
    print("  FEATURE SUMMARY")
    print("=" * 70)

    feat_cols = [c for c in train_features.columns
                 if c not in ("date", "team_1", "team_2", "tournament")]

    groups = {
        "Elo":           [c for c in feat_cols if "elo" in c.lower()],
        "Rolling":       [c for c in feat_cols if any(k in c for k in [
                              "avg_gf", "avg_ga", "avg_gd", "win_rate", "draw_rate", "clean_sheet"
                          ])],
        "H2H":           [c for c in feat_cols if c.startswith("h2h_")],
        "Momentum":      [c for c in feat_cols if any(k in c for k in [
                              "streak", "qualifying", "days_since", "unbeaten", "form_traj"
                          ])],
        "Context":       [c for c in feat_cols if any(k in c for k in [
                              "is_world", "is_wc", "is_cont", "is_friend"
                          ])],
        "Squad":         [c for c in feat_cols if "squad_" in c],
        "Differentials": [c for c in feat_cols if c.startswith("diff_")],
        "Targets":       [c for c in feat_cols if c.startswith("target_")],
    }

    total_listed = 0
    for gname, cols in groups.items():
        if cols:
            print(f"\n  {gname} ({len(cols)}):")
            for c in cols[:8]:
                nn = train_features[c].notna().sum()
                print(f"    • {c:<45s} {nn:>6,} non-null")
            if len(cols) > 8:
                print(f"    ... and {len(cols)-8} more")
            total_listed += len(cols)

    print(f"\n  TOTAL: {total_listed} features + 4 targets")
    print(f"  Training rows:  {len(train_features):,}")
    print(f"  Fixture rows:   {len(fixture_features)}")
    print(f"  Team snapshot:  {len(snapshot)} teams")

    print("\n" + "=" * 70)
    print("  Phase 2 complete. Ready for Phase 3 (modelling).")
    print("=" * 70)

    return train_features, fixture_features, snapshot


if __name__ == "__main__":
    main()
