# src/agents/decider.py
from __future__ import annotations
import json
from typing import Dict
from src.llm.openai_client import OpenAIJSONClient

DECIDER_INSTRUCTIONS = """
You are a senior product data scientist.

Given an experiment design and statistical results, decide:
- ship
- do_not_ship
- inconclusive

Return STRICT JSON with keys:
decision, summary, reasoning, next_steps.
- decision: one of ship / do_not_ship / inconclusive
- reasoning: array of short bullet strings
- next_steps: array of short bullet strings
Do NOT include any markdown report.
Keep strings concise.
"""

def run_decider(design: Dict, stats: Dict, llm: OpenAIJSONClient) -> Dict:
    payload = {
        "design": design,
        "stats": stats,
        "rules_of_thumb": {
            "alpha": 0.05,
            "practical_uplift_abs_min": 0.002
        }
    }
    return llm.json_response(
        instructions=DECIDER_INSTRUCTIONS,
        user_input=json.dumps(payload),
        max_output_tokens=900,
    )
