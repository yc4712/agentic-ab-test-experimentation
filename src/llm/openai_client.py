# src/llm/openai_client.py
# src/llm/openai_client.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict

from openai import OpenAI


@dataclass
class OpenAIJSONClient:
    model: str = "gpt-4.1-mini"

    def __post_init__(self):
        self.client = OpenAI()

    def json_response(
        self,
        *,
        instructions: str,
        user_input: str,
        schema_name: str,
        schema: Dict[str, Any],
        max_output_tokens: int = 800,
    ) -> Dict[str, Any]:
        resp = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=user_input,
            max_output_tokens=max_output_tokens,
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": schema,
                    "strict": True,
                }
            },
        )

        text = (resp.output_text or "").strip()
        if not text:
            raise ValueError("Model returned empty output.")

        return json.loads(text)
