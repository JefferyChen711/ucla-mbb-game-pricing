"""Price elasticity, demand drivers, demand forecast and pricing windows.

1. Elasticity: log(tickets) on log(price) three ways
     naive -> + game controls -> + opponent fixed effects (same opponent, different seasons)
2. Demand drivers: % effect of national TV, conference play, weekend, etc.
3. Demand forecast without price (leave-one-season-out CV) - an honest check of how
   predictable game-level demand is with ~65 games.
4. Revenue sensitivity: what a uniform price change does to revenue given the elasticity.
5. Pricing windows: when buyers purchase and what they pay, by price tier.

Usage:
    python src/analysis.py --games data/processed/games.csv --raw data/raw
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from build_games import channel

BLUE, ORANGE, GRAY, LIGHT, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#8c8c8c", "#c8c7c2", "#0b0b0b", "#52514e", "#e6e5e1"
CONTROLS = "conference_game + rival_usc + weekend + early_tip + national_tv + in_conference_season"
FORECAST_FEATURES = ["conference_game", "rival_usc", "weekend", "early_tip", "national_tv",
                     "in_conference_season", "big_ten", "opp_prev_log_tickets", "opp_prev_missing"]
WINDOWS = ["2+ Months Before", "1-2 Months Before", "2-4 Weeks Before", "1-2 Weeks Before",
           "3-7 Days Before", "1-2 Days Before", "Day Of"]


def style():
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlelocation": "left"})


def elasticity(g, figs):
    rep = g.opponent.map(g.opponent.value_counts()) >= 2
    specs = [
        ("Naive", "lq ~ lp", g),
        ("+ game controls", f"lq ~ lp + {CONTROLS} + C(Season)", g),
        ("+ opponent fixed effects", "lq ~ lp + weekend + early_tip + national_tv + C(Season) + C(opponent)", g[rep]),
    ]
    rows, fits = [], {}
    for name, f, d in specs:
        m = smf.ols(f, d).fit(cov_type="HC1")
        lo, hi = m.conf_int().loc["lp"]
        rows.append({"model": name, "elasticity": m.params["lp"], "ci_low": lo, "ci_high": hi,
                     "p_value": m.pvalues["lp"], "n_games": int(m.nobs), "r2": m.rsquared})
        fits[name] = m
    t = pd.DataFrame(rows)
    print(t.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    y = np.arange(len(t))[::-1]
    for yi, r in zip(y, t.itertuples()):
        c = BLUE if r.model.startswith("+ opp") else GRAY
        ax.plot([r.ci_low, r.ci_high], [yi, yi], color=c, lw=2.5, solid_capstyle="round")
        ax.plot(r.elasticity, yi, "o", color=c, ms=9, mec="white", mew=2)
        ax.text(r.ci_high + 0.08, yi, f"{r.elasticity:+.2f}  (n={r.n_games})", va="center", color=INK, fontsize=10,
                bbox=dict(fc="white", ec="none", pad=1.5), zorder=5)
    ax.axvline(0, color=MUTED, lw=1)
    ax.axvline(-1, color=LIGHT, lw=1, ls="--")
    ax.text(-1.05, y.min() - 0.45, "unit elastic", ha="right", color=MUTED, fontsize=9)
    ax.set_ylim(y.min() - 0.7, y.max() + 0.5)
    ax.set_yticks(y, t.model)
    ax.set_xlim(min(-2.6, t.ci_low.min() - 0.2), max(2.2, t.ci_high.max() + 0.9))
    ax.set_xlabel("Price elasticity of demand (95% CI)")
    ax.set_title("Controlling for opponent flips the sign of the price effect")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    fig.tight_layout(); fig.savefig(figs / "elasticity_estimates.png", dpi=160); plt.close(fig)
    return t, fits["+ game controls"]


def drivers(m, figs):
    labels = {"national_tv": "National TV broadcast", "in_conference_season": "Jan-Mar (conference season)",
              "conference_game": "Conference opponent", "rival_usc": "USC rivalry", "weekend": "Weekend",
              "early_tip": "Tip-off before 5 pm"}
    ci = m.conf_int()
    d = pd.DataFrame({"label": [labels[k] for k in labels],
                      "effect": [np.expm1(m.params[k]) for k in labels],
                      "lo": [np.expm1(ci.loc[k, 0]) for k in labels],
                      "hi": [np.expm1(ci.loc[k, 1]) for k in labels],
                      "p": [m.pvalues[k] for k in labels]}).sort_values("effect")
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    y = np.arange(len(d))
    for yi, r in zip(y, d.itertuples()):
        c = BLUE if r.p < 0.05 else LIGHT
        ax.plot([r.lo * 100, r.hi * 100], [yi, yi], color=c, lw=2.5, solid_capstyle="round")
        ax.plot(r.effect * 100, yi, "o", color=c, ms=9, mec="white", mew=2)
        ax.text(r.hi * 100 + 8, yi, f"{r.effect:+.0%}", va="center", color=INK, fontsize=10)
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(y, d.label); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
    ax.set_xlabel("Effect on primary tickets sold, holding price fixed (%, 95% CI)")
    ax.set_title("What moves demand (blue = significant at 5%)")
    fig.tight_layout(); fig.savefig(figs / "demand_drivers.png", dpi=160); plt.close(fig)
    return d


def add_opponent_history(g):
    """Log tickets the last time UCLA hosted this opponent (strictly earlier game)."""
    g = g.sort_values("game_dt").copy()
    g["opp_prev_log_tickets"] = g.groupby("opponent")["lq"].shift(1)
    g["opp_prev_missing"] = g.opp_prev_log_tickets.isna().astype(int)
    return g


def forecast(g, figs):
    g = add_opponent_history(g)
    seasons = sorted(g.Season.unique())[1:]  # need at least one earlier season to train on
    preds = []
    for s in seasons:
        tr, te = g[g.Season < s].copy(), g[g.Season == s].copy()
        fill = tr.opp_prev_log_tickets.median() if tr.opp_prev_log_tickets.notna().any() else tr.lq.median()
        Xtr = tr[FORECAST_FEATURES].fillna({"opp_prev_log_tickets": fill})
        Xte = te[FORECAST_FEATURES].fillna({"opp_prev_log_tickets": fill})
        ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))).fit(Xtr, tr.lq)
        gbm = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8,
                                        random_state=0).fit(Xtr, tr.lq)
        te["pred_baseline"] = tr.lq.mean()
        te["pred_ridge"] = ridge.predict(Xte)
        te["pred_gbm"] = gbm.predict(Xte)
        preds.append(te)
    p = pd.concat(preds)
    res = {}
    for k in ["baseline", "ridge", "gbm"]:
        err = np.exp(p[f"pred_{k}"]) - p.tickets
        res[k] = {"MAE_tickets": float(np.abs(err).mean()),
                  "median_APE": float(np.median(np.abs(err) / p.tickets)),
                  "corr_log": float(np.corrcoef(p[f"pred_{k}"], p.lq)[0, 1])}
    print(pd.DataFrame(res).T.round(3))
    best = min(["ridge", "gbm"], key=lambda k: res[k]["MAE_tickets"])
    return res, best


def revenue_sensitivity(el):
    """Revenue multiplier for a uniform price change: R1/R0 = (1+dp)^(1+elasticity)."""
    fe = el.iloc[-1]
    rows = []
    for name, e in [("Point estimate", fe.elasticity), ("CI low (more elastic)", fe.ci_low), ("CI high (less elastic)", fe.ci_high)]:
        rows.append({"estimate": name, "elasticity": e,
                     "minus_10pct": 0.9 ** (1 + e) - 1, "plus_10pct": 1.1 ** (1 + e) - 1})
    return pd.DataFrame(rows)


def pricing_windows(raw, g, figs):
    j = pd.read_csv(raw / "mbb_joined_data.csv")
    j = j[(j["Order Qty"] > 0) & (j["Item Payment"] > 0)].copy()
    j = j[channel(j["Price Type Full Name"].fillna("")) == "full_price"]
    tier = g.set_index(["Season", "opponent"]).price_tier
    j["tier"] = [tier.get((s, o), "Unrated") for s, o in zip(j.Season, j.matched_opponent)]
    j = j[j.tier.isin(["Top", "Mid", "Low"]) & j.Purchase_Timing_Bucket.isin(WINDOWS)]
    j["grp"] = np.where(j.tier == "Top", "Top-tier games", "Other games")
    q = j["Order Qty"]
    t = j.assign(rev=j["Item Price"] * q).groupby(["grp", "Purchase_Timing_Bucket"]).agg(tix=("Order Qty", "sum"), rev=("rev", "sum"))
    t["avg_price"] = t.rev / t.tix
    t["share"] = t.tix / t.groupby(level=0).tix.transform("sum")
    t = t.reset_index()

    short = {"2+ Months Before": "2+ mo", "1-2 Months Before": "1-2 mo", "2-4 Weeks Before": "2-4 wk",
             "1-2 Weeks Before": "1-2 wk", "3-7 Days Before": "3-7 d", "1-2 Days Before": "1-2 d", "Day Of": "Day of"}
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    x = np.arange(len(WINDOWS)); w = 0.38
    for i, (grp, c) in enumerate([("Top-tier games", BLUE), ("Other games", GRAY)]):
        d = t[t.grp == grp].set_index("Purchase_Timing_Bucket").reindex(WINDOWS)
        ax[0].bar(x + (i - 0.5) * w, d.share * 100, w - 0.03, color=c, label=grp)
        ax[1].plot(x, d.avg_price, marker="o", color=c, lw=2, label=grp)
        ax[1].text(x[-1] + 0.15, d.avg_price.iloc[-1], grp.split()[0], va="center", color=INK, fontsize=9)
    ax[0].set_title("When full-price tickets are bought (% of tickets)")
    ax[1].set_title("Average full-price ticket price ($)")
    for a in ax:
        a.set_xticks(x, [short[k] for k in WINDOWS], fontsize=9); a.grid(axis="y", color=GRID, lw=0.8)
    ax[0].legend(frameon=False, fontsize=9)
    ax[1].set_ylim(0, None)
    fig.tight_layout(); fig.savefig(figs / "pricing_windows.png", dpi=160); plt.close(fig)
    return t


def main(games, raw, figs, out):
    figs, raw, out = Path(figs), Path(raw), Path(out)
    figs.mkdir(parents=True, exist_ok=True); out.mkdir(parents=True, exist_ok=True)
    style()
    g = pd.read_csv(games, parse_dates=["game_dt"])
    g = g[g.valid & (g.exhibition == 0)].copy()
    g["lq"], g["lp"] = np.log(g.tickets), np.log(g.avg_price)

    el, m_controls = elasticity(g, figs)
    dr = drivers(m_controls, figs)
    fc, best = forecast(g, figs)
    rs = revenue_sensitivity(el)
    win = pricing_windows(raw, pd.read_csv(games), figs)

    el.round(4).to_csv(out / "elasticity_estimates.csv", index=False)
    dr.round(4).to_csv(out / "demand_drivers.csv", index=False)
    rs.round(4).to_csv(out / "revenue_sensitivity.csv", index=False)
    win.round(3).to_csv(out / "pricing_windows.csv", index=False)
    fe = el.iloc[-1]
    summary = {
        "n_games": int(len(g)),
        "elasticity_naive": round(float(el.elasticity.iloc[0]), 2),
        "elasticity_controls": round(float(el.elasticity.iloc[1]), 2),
        "elasticity_opponent_fe": round(float(fe.elasticity), 2),
        "elasticity_opponent_fe_ci": [round(float(fe.ci_low), 2), round(float(fe.ci_high), 2)],
        "forecast_best_model": best,
        "forecast_cv": {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in fc.items()},
        "revenue_change_plus10pct_price": {k: round(float(v), 3) for k, v in rs.set_index("estimate")["plus_10pct"].items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(rs.round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="data/processed/games.csv")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--figs", default="reports/figures")
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()
    main(a.games, a.raw, a.figs, a.out)
