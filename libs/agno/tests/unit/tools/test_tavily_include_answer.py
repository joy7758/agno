"""
BUG #5754 regression guard: web_search_with_tavily must not pass include_answer
to client.get_search_context().

The tavily Python client's get_search_context() does not accept include_answer
as a keyword argument. Passing it raised TypeError.
"""
import ast
from pathlib import Path

AGNO_ROOT = Path(__file__).resolve().parents[3] / "agno"


def test_get_search_context_call_does_not_pass_include_answer():
    """
    BUG #5754: web_search_with_tavily passed include_answer to
    client.get_search_context(), which doesn't accept it.
    """
    tavily_path = AGNO_ROOT / "tools" / "tavily.py"
    source = tavily_path.read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "web_search_with_tavily":
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    func = child.func
                    if (
                        isinstance(func, ast.Attribute)
                        and func.attr == "get_search_context"
                    ):
                        kwarg_names = [kw.arg for kw in child.keywords]
                        assert "include_answer" not in kwarg_names, (
                            "BUG #5754 regression: web_search_with_tavily still passes "
                            "include_answer to get_search_context()."
                        )
            break
    else:
        raise AssertionError("Could not find web_search_with_tavily in tavily.py")
