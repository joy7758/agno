"""
BUG #5329 regression guard: OpenAI/OpenRouter models must map 'reasoning'
attribute to reasoning_content.

When using Gemini models via OpenRouter, the response object has a 'reasoning'
attribute instead of 'reasoning_content'. The parser must check both.
"""
import ast
from pathlib import Path

AGNO_ROOT = Path(__file__).resolve().parents[4] / "agno"


def test_parse_provider_response_checks_reasoning_attribute():
    """
    BUG #5329: _parse_provider_response must check for both
    'reasoning_content' and 'reasoning' attributes on the response message,
    so OpenRouter Gemini responses get reasoning_content populated.
    """
    chat_path = AGNO_ROOT / "models" / "openai" / "chat.py"
    source = chat_path.read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_parse_provider_response":
            # Check that the function body contains a check for '.reasoning'
            source_text = ast.get_source_segment(source, node) or ""
            has_reasoning_check = ".reasoning" in source_text and "reasoning_content" in source_text
            assert has_reasoning_check, (
                "BUG #5329 regression: _parse_provider_response does not check "
                "for 'reasoning' attribute fallback. OpenRouter Gemini models "
                "return reasoning in a 'reasoning' attribute, not 'reasoning_content'."
            )
            return
    raise AssertionError("Could not find _parse_provider_response in chat.py")


def test_parse_provider_response_delta_checks_reasoning_attribute():
    """
    BUG #5329: The streaming variant must also check 'reasoning' fallback.
    """
    chat_path = AGNO_ROOT / "models" / "openai" / "chat.py"
    source = chat_path.read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_parse_provider_response_delta":
            source_text = ast.get_source_segment(source, node) or ""
            has_reasoning_check = ".reasoning" in source_text and "reasoning_content" in source_text
            assert has_reasoning_check, (
                "BUG #5329 regression: _parse_provider_response_delta does not "
                "check for 'reasoning' attribute fallback in streaming."
            )
            return
    raise AssertionError("Could not find _parse_provider_response_delta in chat.py")
