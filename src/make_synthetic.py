"""Synthetic ticket lines + schedule with the same schema as the real files.

The real UCLA Athletics data is confidential and not in this repo. The simulation
builds in the key feature of real ticket pricing: better opponents get HIGHER prices
AND sell more tickets, so a naive price-vs-demand regression is biased upward.
The true elasticity used here is -1.2.

    python src/make_synthetic.py --out data/synthetic
    python src/build_games.py --raw data/synthetic --out data/processed/games_synthetic.csv
    python src/analysis.py --games data/processed/games_synthetic.csv --raw data/synthetic \
        --figs reports/figures_synthetic --out reports/synthetic
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TRUE_ELASTICITY = -1.2
SEASONS = {"MBB21": ("21-22", 2021), "MBB22": ("22-23", 2022), "MBB23": ("23-24", 2023),
           "MBB24": ("24-25", 2024), "MBB25": ("25-26", 2025)}
CONF = [("USC", 1.6), ("Arizona", 1.3), ("Oregon", 0.9), ("Washington", 0.8), ("Stanford", 0.6), ("California", 0.5),
        ("Colorado", 0.5), ("Utah", 0.4), ("Arizona State", 0.7), ("Oregon State", 0.3), ("Washington State", 0.3)]
NONCONF = [(n, q) for n, q in [("Villanova", 1.2), ("Long Beach State", 0.0), ("Pepperdine", 0.1), ("UC Riverside", -0.2),
                                ("CSUN", -0.3), ("Lafayette", -0.4), ("Rider", -0.5), ("Lehigh", -0.5),
                                ("Presbyterian", -0.6), ("Cal Poly", -0.2), ("Denver", -0.3), ("UC Davis", -0.1)]]
WINDOWS = [(-1, 0, "After Game"), (0, 24, "Day Of"), (24, 72, "1-2 Days Before"), (72, 168, "3-7 Days Before"),
           (168, 336, "1-2 Weeks Before"), (336, 720, "2-4 Weeks Before"), (720, 1440, "1-2 Months Before"),
           (1440, 3000, "2+ Months Before")]


def bucket(h):
    for lo, hi, name in WINDOWS:
        if lo <= h < hi:
            return name
    return "2+ Months Before"


def main(out, seed=3):
    rng = np.random.default_rng(seed)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    lines, sheets, cid = [], {}, 30_000_000
    for season, (sheet, year) in SEASONS.items():
        opps = [(o, q, 0) for o, q in rng.permutation(np.array(NONCONF, dtype=object))[:6]] + \
               [(o, q, 1) for o, q in rng.permutation(np.array(CONF, dtype=object))[:9]]
        start = pd.Timestamp(f"{year}-11-04")
        sched = []
        for i, (opp, q, conf) in enumerate(opps):
            q = float(q)
            day = start + pd.Timedelta(days=int(i * 8 + rng.integers(0, 4)))
            tip = pd.Timedelta(hours=int(rng.choice([13, 16, 18, 19, 20])), minutes=int(rng.choice([0, 30])))
            game_dt = day + tip
            tv = rng.random() < 1 / (1 + np.exp(-(1.5 * q - 0.3)))
            conf_season = game_dt.month in (1, 2, 3)
            # price is SET from opponent quality (+ season-to-season noise): the confounder
            price = float(np.exp(3.6 + 0.45 * q + rng.normal(0, 0.18)))
            log_q = 5.9 + 1.3 * q + 0.5 * tv + 0.45 * conf_season + TRUE_ELASTICITY * (np.log(price) - 3.6) + rng.normal(0, 0.35)
            n_tix = int(np.exp(log_q))
            sched.append([game_dt.strftime("%B %d, %Y"), game_dt.strftime("%I:%M %p (PT)"), "Home", opp,
                          "Los Angeles, Calif. | Pauley Pavilion", "ESPN" if tv else "BTN", "AM 790",
                          f"{'W' if rng.random() < .7 else 'L'}, 75-68", "Links"])
            sold = 0
            while sold < n_tix:
                qty = int(rng.choice([1, 2, 2, 2, 3, 4]))
                hours = float(rng.exponential(400 if q > 0.8 else 120))
                lines.append({"Season": season, "Customer ID": cid, "Item": f"UCLA Men's Basketball vs {opp}"[:30],
                              "matched_opponent": opp, "Orig Date-Time": (game_dt - pd.Timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S"),
                              "game_dt": game_dt.strftime("%Y-%m-%d %H:%M:%S"), "hours_before_game": hours,
                              "Purchase_Timing_Bucket": bucket(hours), "Order Qty": qty,
                              "Item Price": round(price * rng.uniform(0.6, 1.4) * (1 + min(hours, 2000) / 4000)),
                              "Price Type Code": "SYN", "Price Type Full Name": "Regular Price"})
                lines[-1]["Item Payment"] = lines[-1]["Item Price"] * qty
                sold += qty; cid += 1
            for _ in range(int(rng.integers(300, 700))):  # non-primary activity (pass claims, resale)
                ptype = rng.choice(["Claim Fast Break Pass Ticket", "STUBHUB - Season Ticket"])
                hours = float(rng.exponential(48))
                pay = 0 if ptype.startswith("Claim") else round(price * rng.uniform(0.7, 1.5))
                lines.append({"Season": season, "Customer ID": cid, "Item": f"UCLA Men's Basketball vs {opp}"[:30],
                              "matched_opponent": opp, "Orig Date-Time": (game_dt - pd.Timedelta(hours=hours)).strftime("%Y-%m-%d %H:%M:%S"),
                              "game_dt": game_dt.strftime("%Y-%m-%d %H:%M:%S"), "hours_before_game": hours,
                              "Purchase_Timing_Bucket": bucket(hours), "Order Qty": 1, "Item Price": pay,
                              "Price Type Code": "SYN", "Price Type Full Name": ptype, "Item Payment": pay})
                cid += 1
        sheets[sheet] = pd.DataFrame(sched)
        print(season, len(opps), "games")
    pd.DataFrame(lines).to_csv(out / "mbb_joined_data.csv", index=False)
    with pd.ExcelWriter(out / "mbb schedule.xlsx") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, header=False, index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    main(ap.parse_args().out)
