from src.llm.openai_client import _extract_first_json_object

def test_extract_first_json_object_from_messy_text():
text = 'Hello\n{"a": 1, "b": {"c": 2}}\nThanks'
extracted = _extract_first_json_object(text)
assert extracted == '{"a": 1, "b": {"c": 2}}'
