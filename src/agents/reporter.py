from __future__ import annotations
import json
from typing import Dict
from src.llm.openai_client import OpenAIJSONClient

REPORTER_INSTRUCTIONS = """
Write a concise markdown experiment report for a product analytics audience.
Use the provided design + stats + decision.
Return ONLY markdown (no JSON).
Keep it under 500 words.
"""

def run_reporter(design: Dict, stats: Dict, decision: Dict, llm: OpenAIJSONClient) -> str:
    payload = {"design": design, "stats": stats, "decision": decision}
    resp = llm.client.responses.create(
        model=llm.model,
        instructions=REPORTER_INSTRUCTIONS,
        input=json.dumps(payload),
        max_output_tokens=700,
    )
    return (resp.output_text or "").strip()
