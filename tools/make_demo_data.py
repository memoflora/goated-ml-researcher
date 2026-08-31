"""Generate a small synthetic dataset so the agnostic path can be demonstrated offline.

    python tools/make_demo_data.py

Writes `data/demo-regression/listings.csv` — a tabular rent-prediction problem with
nothing whatsoever in common with KuaiRand: a continuous target, mixed numeric and
categorical features, real missingness, a constant column and one deliberately leaky
column. The last two exist so the profiler's warnings have something to catch.

This is a *demo fixture*, not competition data. It never touches the KuaiRand benchmark and
is not training data for anything we submit.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "demo-regression"
N = 12_000
SEED = 0


def main() -> int:
    rng = np.random.default_rng(SEED)

    neighbourhood = rng.choice(
        ["riverside", "old-town", "docks", "university", "hillside"],
        size=N,
        p=[0.28, 0.22, 0.18, 0.20, 0.12],
    )
    prop_type = rng.choice(["flat", "house", "studio", "maisonette"], size=N, p=[0.5, 0.25, 0.18, 0.07])
    bedrooms = rng.integers(1, 6, size=N)
    bathrooms = np.clip(rng.poisson(1.2, size=N), 1, 4)
    area_m2 = np.round(28 + 22 * bedrooms + rng.normal(0, 9, size=N), 1).clip(18, None)
    floor = rng.integers(0, 12, size=N)
    year_built = rng.integers(1890, 2024, size=N)
    has_lift = (rng.random(N) < 0.42).astype(int)
    has_garden = ((prop_type == "house") & (rng.random(N) < 0.7)).astype(int)
    distance_km = np.round(np.abs(rng.normal(4.5, 2.6, size=N)), 2)
    epc = rng.choice(list("ABCDEFG"), size=N, p=[0.04, 0.12, 0.26, 0.30, 0.16, 0.08, 0.04])

    hood_effect = {
        "riverside": 340.0,
        "old-town": 280.0,
        "docks": 120.0,
        "university": 190.0,
        "hillside": 95.0,
    }
    type_effect = {"flat": 0.0, "house": 210.0, "studio": -140.0, "maisonette": 95.0}

    rent = (
        420
        + 7.4 * area_m2
        + 96 * bedrooms
        + 58 * bathrooms
        + np.array([hood_effect[h] for h in neighbourhood])
        + np.array([type_effect[t] for t in prop_type])
        - 34 * distance_km
        + 61 * has_lift
        + 88 * has_garden
        + 0.9 * (year_built - 1950)
        + rng.normal(0, 120, size=N)
    )
    rent = np.round(np.clip(rent, 250, None), 2)

    # Missingness that a pipeline has to actually handle.
    epc = epc.astype(object)
    epc[rng.random(N) < 0.09] = None
    floor_f = floor.astype(float)
    floor_f[rng.random(N) < 0.05] = np.nan

    rows = {
        "listing_id": np.arange(N),
        "neighbourhood": neighbourhood,
        "property_type": prop_type,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "area_m2": area_m2,
        "floor": floor_f,
        "year_built": year_built,
        "has_lift": has_lift,
        "has_garden": has_garden,
        "distance_to_centre_km": distance_km,
        "epc_rating": epc,
        # Constant — the profiler should say so.
        "listing_currency": np.full(N, "GBP"),
        # Leaky — a near-copy of the target. The profiler should flag it, and a good
        # agent should notice the warning rather than joyfully using it.
        "agent_valuation": np.round(rent * 1.0004 + rng.normal(0, 0.4, size=N), 2),
        "monthly_rent": rent,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "listings.csv"
    header = ",".join(rows)
    with open(out, "w", encoding="utf-8", newline="") as fh:
        fh.write(header + "\n")
        cols = list(rows.values())
        for i in range(N):
            fh.write(
                ",".join("" if (v := c[i]) is None or (isinstance(v, float) and np.isnan(v))
                         else str(v) for c in cols)
                + "\n"
            )
    print(f"wrote {out} — {N:,} rows, {len(rows)} columns")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
