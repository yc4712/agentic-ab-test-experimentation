# src/run_demo.py
from __future__ import annotations
import os
import json
from dotenv import load_dotenv

from src.llm.openai_client import OpenAIJSONClient
from src.agents.designer import run_designer
from src.agents.analyst import run_analyst
from src.agents.decider import run_decider
from src.agents.reporter import run_reporter

def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set. Put it in .env (not committed).")

    description = (
        "We tested a new feed ranking layout to increase conversion rate "
        "(e.g., click-through to the next video)."
    )
    csv_path = "data/sample_experiment.csv"

    llm = OpenAIJSONClient(model="gpt-5.2")

    design = run_designer(description, llm)
    stats = run_analyst(csv_path)
    decision = run_decider(design, stats, llm)
    report_md = run_reporter(design, stats, decision, llm)

    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/design.json", "w", encoding="utf-8") as f:
        json.dump(design, f, indent=2)

    with open("artifacts/stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    with open("artifacts/decision.json", "w", encoding="utf-8") as f:
        json.dump(decision, f, indent=2)

    with open("artifacts/report.md", "w", encoding="utf-8") as f:
        f.write(report_md)

    print("\n=== DECISION ===")
    print(decision["decision"])
    print("\n=== SUMMARY ===")
    print(decision["summary"])
    print("\nPlot saved to:", stats.get("plot_path"))
    reasoning = decision.get("reasoning")
    if isinstance(reasoning, list) and reasoning:
        print("\n=== REASONING ===:")
        for r in reasoning:
            print(f" - {r}")
    print("\n=== NEXT STEPS ===")
    print(decision["next_steps"])

if __name__ == "__main__":
    main()
