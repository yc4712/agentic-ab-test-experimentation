# src/llm/openai_client.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from openai import OpenAI


class LLMResponseError(RuntimeError):
    pass


class LLMRefusalError(LLMResponseError):
    pass


class LLMIncompleteError(LLMResponseError):
    pass


@dataclass
class OpenAIJSONClient:
    model: str = "gpt-4.1-mini"
    client: Optional[Any] = field(default=None, init=False, repr=False)

    def _get_client(self):
        if self.client is None:
            self.client = OpenAI()
        return self.client

    def json_response(
        self,
        *,
        instructions: str,
        user_input: str,
        schema_name: str,
        schema: Dict[str, Any],
        max_output_tokens: int = 800,
    ) -> Dict[str, Any]:
        resp = self._get_client().responses.create(
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

        if getattr(resp, "status", None) == "incomplete":
            details = getattr(resp, "incomplete_details", None)
            reason = getattr(details, "reason", "unknown")
            raise LLMIncompleteError(f"Response incomplete: {reason}")

        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", None) != "message":
                continue
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", None) == "refusal":
                    msg = getattr(content, "refusal", "Model refused request.")
                    raise LLMRefusalError(msg)

        text = (getattr(resp, "output_text", None) or "").strip()
        if not text:
            raise LLMResponseError("Model returned no structured text.")

        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMResponseError(f"Structured output was not valid JSON: {e}") from e
