# src/agents/designer.py
from __future__ import annotations
from typing import Dict
from src.llm.openai_client import OpenAIJSONClient

DESIGNER_INSTRUCTIONS = """
You are a senior experimentation scientist.
Given a short product experiment description, produce a structured experiment plan.
Return JSON that matches the provided schema exactly.
"""

DESIGN_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "hypothesis": {"type": "string"},
        "primary_metric": {"type": "string"},
        "unit_of_randomization": {"type": "string"},
        "expected_direction": {"type": "string"},
        "risks": {"type": "array", "items": {"type": "string"}},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "hypothesis",
        "primary_metric",
        "unit_of_randomization",
        "expected_direction",
        "risks",
        "assumptions",
        "notes",
    ],
}


def run_designer(description: str, llm: OpenAIJSONClient) -> Dict:
    return llm.json_response(
        instructions=DESIGNER_INSTRUCTIONS,
        user_input=f"Experiment description:\n{description}",
        schema_name="experiment_design",
        schema=DESIGN_SCHEMA,
        max_output_tokens=500,
    )
