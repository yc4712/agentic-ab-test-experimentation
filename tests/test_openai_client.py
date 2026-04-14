# tests/test_openai_client.py
from types import SimpleNamespace

from src.llm.openai_client import OpenAIJSONClient


class FakeResponses:
    def __init__(self, output_text: str):
        self._output_text = output_text

    def create(self, **kwargs):
        return SimpleNamespace(output_text=self._output_text)


class FakeOpenAI:
    def __init__(self, output_text: str):
        self.responses = FakeResponses(output_text)


def test_json_response_parses_valid_schema_output():
    client = OpenAIJSONClient(model="test-model")
    client.client = FakeOpenAI('{"hypothesis":"x","primary_metric":"ctr","unit_of_randomization":"user","expected_direction":"up","risks":[],"assumptions":[],"notes":[]}')
    result = client.json_response(
        instructions="test",
        user_input="test",
        schema_name="experiment_design",
        schema={
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
        },
    )
    assert result["primary_metric"] == "ctr"


def test_json_response_raises_on_empty_output():
    client = OpenAIJSONClient(model="test-model")
    client.client = FakeOpenAI("")
    try:
        client.json_response(
            instructions="test",
            user_input="test",
            schema_name="x",
            schema={"type": "object", "properties": {}, "required": [], "additionalProperties": False},
        )
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "empty output" in str(e).lower()
