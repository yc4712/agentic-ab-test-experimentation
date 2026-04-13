from src.tools.plots import plot_conversion_rates

def test_plot_conversion_rates_writes_file(tmp_path):
    stats = {
        "variants": {
            "A": {"conversion_rate": 0.10},
            "B": {"conversion_rate": 0.12},
        }
    }

    output = tmp_path / "plot.png"
    path = plot_conversion_rates(stats, str(output))

    assert output.exists()
    assert path.endswith(".png")
