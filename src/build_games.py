"""Aggregate single-game ticket lines to one row per home game and join the schedule.

Demand measure: PRIMARY PAID tickets = tickets UCLA sold directly at a price > $0
(full price, promo and group). Resale, transfers, pass claims and season-seat
inventory are excluded because UCLA does not set those prices game by game.

Usage:
    python src/build_games.py --raw data/raw --out data/processed/games.csv
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

CHANNEL_RULES = [  # same channel mapping as the buyer-segmentation project
    ("resale", r"STUBHUB|SEATGEEK|^SH -"),
    ("transfer", r"^TR -"),
    ("pass_claim", r"Claim Fast Break|Access Pass|Den Pass"),
    ("group", r"Group|Career Fair|Athletics For Youth|LA County|Visiting Team"),
    ("promo", r"Fevo|Offer|Flash Sale|Friday|Monday|Appreciation|Student For A Day|Mini|Pick a Pair|"
              r"3-Pointer|Package|Weekend|Recognition|Dynamic Pricing Partners|Men of Westwood"),
    ("season_seat", r"Season|STX"),
]
PRIMARY = {"full_price", "promo", "group"}
NATIONAL_TV = {"ESPN", "ESPN2", "FS1", "FOX", "CBS", "PEACOCK"}
SEASON_SHEETS = {"21-22": "MBB21", "22-23": "MBB22", "23-24": "MBB23", "24-25": "MBB24", "25-26": "MBB25"}
BIG_TEN_SEASONS = {"MBB24", "MBB25"}  # UCLA moved from the Pac-12 to the Big Ten in 2024-25
PAC_12 = {"Arizona", "Arizona State", "California", "Colorado", "Oregon", "Oregon State", "Stanford", "USC",
          "Utah", "Washington", "Washington State"}
BIG_TEN = {"Illinois", "Indiana", "Iowa", "Maryland", "Michigan", "Michigan State", "Minnesota", "Nebraska",
           "Northwestern", "Ohio State", "Oregon", "Penn State", "Purdue", "Rutgers", "USC", "Washington", "Wisconsin"}


def channel(pt: pd.Series) -> pd.Series:
    out = pd.Series("full_price", index=pt.index)
    done = pd.Series(False, index=pt.index)
    for name, pat in CHANNEL_RULES:
        m = pt.str.contains(pat, case=False, regex=True) & ~done
        out[m], done = name, done | m
    return out


def load_schedule(path: Path) -> pd.DataFrame:
    """Return Season, date, tv, result for each home game (sheet layouts differ by season)."""
    rows = []
    for sheet, df in pd.read_excel(path, sheet_name=None, header=None).items():
        season = SEASON_SHEETS.get(sheet.strip())
        if season is None:
            continue
        for r in df.itertuples(index=False):
            vals = [v for v in r if not (isinstance(v, float) and np.isnan(v))]
            date = pd.to_datetime(str(r[0]).replace(",", ""), errors="coerce")
            if pd.isna(date):
                continue
            tv = next((str(v).strip() for v in vals if str(v).strip().upper() in NATIONAL_TV | {
                "PAC-12 NETWORKS", "BTN", "B1G+", "UCLA LIVE STREAM"}), "")
            res = next((str(v) for v in vals if re.match(r"^[WL],\s*\d", str(v))), "")
            rows.append({"Season": season, "date": date.normalize(), "tv": tv, "result": res[:1]})
    return pd.DataFrame(rows)


def build(raw: Path) -> pd.DataFrame:
    j = pd.read_csv(raw / "mbb_joined_data.csv")
    j = j[(j["Order Qty"] > 0) & j["matched_opponent"].notna()].copy()
    j["game_dt"] = pd.to_datetime(j["game_dt"])
    j["order_dt"] = pd.to_datetime(j["Orig Date-Time"])
    j["channel"] = channel(j["Price Type Full Name"].fillna(""))
    j["primary_paid"] = j["channel"].isin(PRIMARY) & (j["Item Payment"] > 0)
    q = j["Order Qty"]
    j["pp_tix"] = q * j.primary_paid
    j["pp_rev"] = j["Item Payment"] * j.primary_paid
    j["early_pp_tix"] = j.pp_tix * (j["hours_before_game"] >= 24 * 14)

    g = j.groupby(["Season", "game_dt", "matched_opponent"]).agg(
        tickets_all=("Order Qty", "sum"), tickets=("pp_tix", "sum"), revenue=("pp_rev", "sum"),
        early_tickets=("early_pp_tix", "sum"),
    ).reset_index().rename(columns={"matched_opponent": "opponent"})
    g["avg_price"] = g.revenue / g.tickets.replace(0, np.nan)
    g["share_bought_early"] = g.early_tickets / g.tickets.replace(0, np.nan)

    # Data-quality filters -------------------------------------------------------
    cutoff = j.groupby("Season")["order_dt"].max()  # last order in each season's export
    g["sales_complete"] = g.apply(lambda r: r.game_dt <= cutoff[r.Season], axis=1)
    g["valid"] = g.sales_complete & (g.tickets_all >= 500) & (g.tickets >= 50)

    # Game attributes -------------------------------------------------------------
    g["exhibition"] = g.opponent.str.contains("exhibition|at UCLA", case=False).astype(int)
    # Price tier (descriptive only - it is built from price, so never used as a control for price)
    opp_price = g[g.valid & (g.exhibition == 0)].groupby("opponent").avg_price.mean()
    hi, mid = opp_price.quantile(0.75), opp_price.quantile(0.40)
    op = g.opponent.map(opp_price)
    g["price_tier"] = np.select([g.exhibition == 1, op.isna(), op >= hi, op >= mid],
                                ["Exhibition", "Unrated", "Top", "Mid"], "Low")
    # Opponent quality proxies that do NOT come from price
    g["conference_game"] = [int(o in (BIG_TEN if s in BIG_TEN_SEASONS else PAC_12)) for o, s in zip(g.opponent, g.Season)]
    g["rival_usc"] = (g.opponent == "USC").astype(int)
    g["weekend"] = (g.game_dt.dt.day_of_week >= 5).astype(int)
    g["tip_hour"] = g.game_dt.dt.hour + g.game_dt.dt.minute / 60
    g["early_tip"] = (g.tip_hour < 17).astype(int)
    g["month"] = g.game_dt.dt.month
    g["in_conference_season"] = g.month.isin([1, 2, 3]).astype(int)
    g["big_ten"] = g.Season.isin(BIG_TEN_SEASONS).astype(int)

    sched_path = raw / "mbb schedule.xlsx"
    if sched_path.exists():
        s = load_schedule(sched_path)
        g["date"] = g.game_dt.dt.normalize()
        g = g.merge(s, on=["Season", "date"], how="left").drop(columns="date")
    else:
        g["tv"], g["result"] = "", ""
    g["national_tv"] = g.tv.fillna("").str.upper().isin(NATIONAL_TV).astype(int)
    return g.sort_values("game_dt").reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="data/processed/games.csv")
    a = ap.parse_args()
    g = build(Path(a.raw))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    g.to_csv(a.out, index=False)
    print(g.groupby("Season").agg(games=("opponent", "size"), valid=("valid", "sum"),
                                  tickets=("tickets", "sum"), revenue=("revenue", "sum")))
    print(g.price_tier.value_counts())
