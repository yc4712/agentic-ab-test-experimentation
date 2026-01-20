# src/tools/plots.py
from __future__ import annotations
import os
import matplotlib.pyplot as plt

def plot_conversion_rates(stats: dict, save_path: str = "plots/conv_rates.png") -> str:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    variants = list(stats["variants"].keys())
    rates = [stats["variants"][v]["conversion_rate"] for v in variants]

    plt.figure()
    plt.bar(variants, rates)
    plt.ylabel("Conversion rate")
    plt.title("Conversion rate by variant")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    return save_path
