"""
BUG #5173 regression guard: Milvus search() and async_search() must accept
search_params for radius/range_filter support.
"""
import ast
from pathlib import Path

AGNO_ROOT = Path(__file__).resolve().parents[3] / "agno"


def test_milvus_search_accepts_search_params():
    """
    BUG #5173: Milvus.search() must accept search_params kwarg.
    """
    milvus_path = AGNO_ROOT / "vectordb" / "milvus" / "milvus.py"
    source = milvus_path.read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Milvus":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "search":
                    params = [arg.arg for arg in item.args.args + item.args.kwonlyargs]
                    assert "search_params" in params, (
                        "BUG #5173 regression: Milvus.search() does not accept "
                        "search_params. Users cannot pass radius/range_filter."
                    )
                    return
    raise AssertionError("Could not find Milvus.search in milvus.py")


def test_milvus_async_search_accepts_search_params():
    """
    BUG #5173: Milvus.async_search() must also accept search_params.
    """
    milvus_path = AGNO_ROOT / "vectordb" / "milvus" / "milvus.py"
    source = milvus_path.read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Milvus":
            for item in node.body:
                if isinstance(item, ast.AsyncFunctionDef) and item.name == "async_search":
                    params = [arg.arg for arg in item.args.args + item.args.kwonlyargs]
                    assert "search_params" in params, (
                        "BUG #5173 regression: Milvus.async_search() does not "
                        "accept search_params."
                    )
                    return
    raise AssertionError("Could not find Milvus.async_search in milvus.py")
