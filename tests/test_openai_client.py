# tests/test_openai_client.py
from types import SimpleNamespace

import pytest

from src.llm.openai_client import (
    LLMIncompleteError,
    LLMRefusalError,
    LLMResponseError,
    OpenAIJSONClient,
)


class FakeResponses:
    def __init__(self, response):
        self._response = response

    def create(self, **kwargs):
        return self._response


class FakeOpenAI:
    def __init__(self, response):
        self.responses = FakeResponses(response)


def make_client(response):
    client = OpenAIJSONClient(model="test-model")
    client.client = FakeOpenAI(response)
    return client


SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {"x": {"type": "string"}},
    "required": ["x"],
}


def test_json_response_success():
    resp = SimpleNamespace(
        status="completed",
        output=[],
        output_text='{"x":"ok"}',
    )
    client = make_client(resp)
    result = client.json_response(
        instructions="test",
        user_input="test",
        schema_name="test_schema",
        schema=SCHEMA,
    )
    assert result == {"x": "ok"}


def test_json_response_incomplete():
    resp = SimpleNamespace(
        status="incomplete",
        incomplete_details=SimpleNamespace(reason="max_output_tokens"),
        output=[],
        output_text="",
    )
    client = make_client(resp)
    with pytest.raises(LLMIncompleteError):
        client.json_response(
            instructions="test",
            user_input="test",
            schema_name="test_schema",
            schema=SCHEMA,
        )


def test_json_response_refusal():
    resp = SimpleNamespace(
        status="completed",
        output=[
            SimpleNamespace(
                type="message",
                content=[SimpleNamespace(type="refusal", refusal="Cannot comply")],
            )
        ],
        output_text="",
    )
    client = make_client(resp)
    with pytest.raises(LLMRefusalError):
        client.json_response(
            instructions="test",
            user_input="test",
            schema_name="test_schema",
            schema=SCHEMA,
        )


def test_json_response_empty_output():
    resp = SimpleNamespace(
        status="completed",
        output=[],
        output_text="",
    )
    client = make_client(resp)
    with pytest.raises(LLMResponseError):
        client.json_response(
            instructions="test",
            user_input="test",
            schema_name="test_schema",
            schema=SCHEMA,
        )
