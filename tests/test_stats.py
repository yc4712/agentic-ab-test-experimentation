import pandas as pd
import pytest

from src.tools.stats import compute_ab_stats, load_experiment_data


def test_compute_ab_stats_basic():
    df = pd.DataFrame(
        {
            "variant": ["A", "A", "A", "B", "B", "B"],
            "converted": [0, 1, 0, 1, 1, 0],
        }
    )

    stats = compute_ab_stats(df)

    assert "variants" in stats
    assert "uplift_abs" in stats
    assert "p_value" in stats
    assert stats["variants"]["A"]["n"] == 3
    assert stats["variants"]["B"]["n"] == 3


def test_compute_ab_stats_requires_a_and_b():
    df = pd.DataFrame(
        {
            "variant": ["A", "A", "C"],
            "converted": [0, 1, 1],
        }
    )

    with pytest.raises(ValueError):
        compute_ab_stats(df)


def test_load_experiment_data_requires_columns(tmp_path):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text("variant,clicks\nA,1\nB,0\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_experiment_data(str(csv_path))
