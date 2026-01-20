# src/agents/analyst.py
from __future__ import annotations
from typing import Dict
from src.tools.stats import load_experiment_data, compute_ab_stats
from src.tools.plots import plot_conversion_rates

def run_analyst(csv_path: str) -> Dict:
    df = load_experiment_data(csv_path)
    stats = compute_ab_stats(df)
    stats["plot_path"] = plot_conversion_rates(stats, "plots/conv_rates.png")
    return stats
