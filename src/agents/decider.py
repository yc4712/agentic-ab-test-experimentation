# src/agents/decider.py
from __future__ import annotations
import json
from typing import Dict
from src.llm.openai_client import OpenAIJSONClient

DECIDER_INSTRUCTIONS = """
You are a senior product data scientist.
Decide whether the experiment outcome is ship, do_not_ship, or inconclusive.
Return JSON that matches the provided schema exactly.
"""

DECISION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "decision": {
            "type": "string",
            "enum": ["ship", "do_not_ship", "inconclusive"],
        },
        "summary": {"type": "string"},
        "reasoning": {"type": "array", "items": {"type": "string"}},
        "next_steps": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["decision", "summary", "reasoning", "next_steps"],
}


def run_decider(design: Dict, stats: Dict, llm: OpenAIJSONClient) -> Dict:
    payload = {
        "design": design,
        "stats": stats,
        "rules_of_thumb": {
            "alpha": 0.05,
            "practical_uplift_abs_min": 0.002,
        },
    }
    return llm.json_response(
        instructions=DECIDER_INSTRUCTIONS,
        user_input=json.dumps(payload),
        schema_name="experiment_decision",
        schema=DECISION_SCHEMA,
        max_output_tokens=700,
    )
