# src/agents/designer.py
from __future__ import annotations
from typing import Dict
from src.llm.openai_client import OpenAIJSONClient

DESIGNER_INSTRUCTIONS = """
You are a senior experimentation scientist.
Given a short product experiment description, produce a structured experiment plan.

Return STRICT JSON with keys:
hypothesis, primary_metric, unit_of_randomization, expected_direction,
risks, assumptions, notes.
"""

def run_designer(description: str, llm: OpenAIJSONClient) -> Dict:
    return llm.json_response(
        instructions=DESIGNER_INSTRUCTIONS,
        user_input=f"Experiment description:\n{description}",
        max_output_tokens=600
    )
