# src/tools/io.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd

def simulate_ab_csv(
    out_path: str,
    n_users: int = 20000,
    p_a: float = 0.10,
    p_b: float = 0.115,
    seed: int = 42,
) -> str:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rng = np.random.default_rng(seed)

    user_id = np.arange(1, n_users + 1)
    variant = rng.choice(["A", "B"], size=n_users)
    p = np.where(variant == "A", p_a, p_b)
    converted = (rng.random(n_users) < p).astype(int)

    df = pd.DataFrame(
        {"user_id": user_id, "variant": variant, "converted": converted}
    )
    df.to_csv(out_path, index=False)
    return out_path
