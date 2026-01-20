# src/llm/openai_client.py
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional

from openai import OpenAI


def _extract_first_json_object(text: str) -> str:
    """
    Best-effort extraction of the first {...} JSON object from text.
    Handles extra prose before/after.
    """
    # Find the first '{' and then scan forward counting braces
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object start '{' found in model output.")

    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]

    raise ValueError("Found '{' but did not find matching '}' (truncated output?).")


@dataclass
class OpenAIJSONClient:
    model: str = "gpt-4.1-mini"  # use a real model name you have access to

    def __post_init__(self):
        self.client = OpenAI()

    def json_response(
        self,
        *,
        instructions: str,
        user_input: str,
        max_output_tokens: int = 800,
        debug_dump_path: str = "artifacts/llm_last_output.txt",
    ) -> Dict[str, Any]:
        os.makedirs(os.path.dirname(debug_dump_path), exist_ok=True)

        resp = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=user_input,
            max_output_tokens=max_output_tokens
        )
        text = (resp.output_text or "").strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Dump raw output for you to inspect
            with open(debug_dump_path, "w", encoding="utf-8") as f:
                f.write(text)

            # Attempt to extract the first JSON object if extra text exists
            try:
                extracted = _extract_first_json_object(text)
                return json.loads(extracted)
            except Exception:
                pass

            # One deterministic "repair" attempt
            fix = self.client.responses.create(
                model=self.model,
                instructions=(
                    "Return ONLY valid JSON. No prose, no markdown. "
                    "Ensure strings are properly escaped."
                ),
                input=f"Convert this to valid JSON only:\n\n{text}",
                max_output_tokens=max_output_tokens,
            )
            fixed_text = (fix.output_text or "").strip()

            # Dump repaired output too (in case it still fails)
            with open(debug_dump_path, "a", encoding="utf-8") as f:
                f.write("\n\n--- FIX ATTEMPT OUTPUT ---\n")
                f.write(fixed_text)

            # Try parse, with extraction fallback
            try:
                return json.loads(fixed_text)
            except json.JSONDecodeError:
                extracted = _extract_first_json_object(fixed_text)
                return json.loads(extracted)
