"""
BUG #5858 regression guard: Crawl4aiTools must accept proxy_config.

Users had no way to configure a proxy for Crawl4AI web crawling.
"""
import ast
from pathlib import Path

AGNO_ROOT = Path(__file__).resolve().parents[3] / "agno"


def test_crawl4ai_accepts_proxy_config():
    """
    BUG #5858: Crawl4aiTools.__init__ must accept proxy_config parameter.
    """
    crawl4ai_path = AGNO_ROOT / "tools" / "crawl4ai.py"
    source = crawl4ai_path.read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Crawl4aiTools":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    params = [arg.arg for arg in item.args.args + item.args.kwonlyargs]
                    assert "proxy_config" in params, (
                        "BUG #5858 regression: Crawl4aiTools.__init__ does not "
                        "accept proxy_config."
                    )
                    return
    raise AssertionError("Could not find Crawl4aiTools.__init__ in crawl4ai.py")


def test_crawl4ai_stores_proxy_config():
    """
    BUG #5858: proxy_config must be stored as self.proxy_config.
    """
    crawl4ai_path = AGNO_ROOT / "tools" / "crawl4ai.py"
    source = crawl4ai_path.read_text()
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Crawl4aiTools":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    for stmt in ast.walk(item):
                        if (
                            isinstance(stmt, ast.Assign)
                            and len(stmt.targets) == 1
                            and isinstance(stmt.targets[0], ast.Attribute)
                            and stmt.targets[0].attr == "proxy_config"
                        ):
                            return
    raise AssertionError(
        "BUG #5858 regression: Crawl4aiTools.__init__ does not store "
        "self.proxy_config."
    )
