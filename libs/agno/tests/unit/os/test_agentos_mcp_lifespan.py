"""
BUG #5334 regression guard: AgentOS must not wrap MCP lifespan twice.

On AWS Lambda (stateless), the double MCP lifespan wrapping caused the
MCP session to init twice, leading to connection errors. Fixed by removing
duplicate lifespan wrapping in _make_app.
"""
import ast
import inspect

import agno.os.app as app_module


def test_make_app_does_not_add_mcp_lifespan():
    """
    BUG #5334: AgentOS._make_app() must NOT add mcp_lifespan itself.
    MCP lifespan is already added in the caller (get_app/aget_app).
    If _make_app also adds it, it wraps twice, causing double-init on Lambda.
    """
    source = inspect.getsource(app_module)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "_make_app":
            func_source = ast.dump(node)
            assert "mcp_lifespan" not in func_source, (
                "BUG #5334 regression: _make_app references mcp_lifespan. "
                "MCP lifespan should only be added in get_app/aget_app, not in "
                "_make_app, to avoid double-wrapping on stateless platforms like Lambda."
            )
            break
    else:
        raise AssertionError("Could not find _make_app in app.py")
