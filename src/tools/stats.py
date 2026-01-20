# src/tools/stats.py
from __future__ import annotations
import pandas as pd
import numpy as np
from scipy.stats import norm

def load_experiment_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {"variant", "converted"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # basic cleanup
    df = df.copy()
    df["variant"] = df["variant"].astype(str)
    df["converted"] = df["converted"].astype(int)
    return df

def compute_ab_stats(df: pd.DataFrame) -> dict:
    # assumes binary outcome in converted and variants A/B
    grouped = df.groupby("variant")["converted"]

    variants = {}
    for v, s in grouped:
        variants[v] = {"n": int(s.shape[0]), "conversion_rate": float(s.mean())}

    if "A" not in variants or "B" not in variants:
        raise ValueError(f"Expected variants A and B, got {sorted(variants.keys())}")

    n_a = variants["A"]["n"]
    n_b = variants["B"]["n"]
    p_a = variants["A"]["conversion_rate"]
    p_b = variants["B"]["conversion_rate"]

    # pooled z-test for difference in proportions
    p_pool = (p_a * n_a + p_b * n_b) / (n_a + n_b)
    se_pool = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    if se_pool == 0:
        z = float("nan")
        p_value = float("nan")
        ci_95 = [float("nan"), float("nan")]
    else:
        z = (p_b - p_a) / se_pool
        p_value = float(2 * (1 - norm.cdf(abs(z))))
        zcrit = float(norm.ppf(0.975))
        diff = p_b - p_a
        ci_95 = [float(diff - zcrit * se_pool), float(diff + zcrit * se_pool)]

    uplift_abs = float(p_b - p_a)
    uplift_rel = float(uplift_abs / p_a) if p_a > 0 else float("nan")

    return {
        "variants": variants,
        "uplift_abs": uplift_abs,
        "uplift_rel": uplift_rel,
        "z_score": float(z),
        "p_value": p_value,
        "ci_95": ci_95,
    }
