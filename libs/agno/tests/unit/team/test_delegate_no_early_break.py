"""
BUG #4688 regression guard: delegate_task_to_member must not break async iterator.

Breaking out of an async for loop triggers GeneratorExit on the underlying
generator, which caused connection leaks and session save failures in AgentOS.
The fix was to use continue instead of break, letting the iterator exhaust
naturally.
"""
import ast
import inspect


def test_adelegate_task_to_member_does_not_break_iterator():
    """
    BUG #4688: adelegate_task_to_member used 'break' inside 'async for'
    over the member's response stream, triggering GeneratorExit. The fix
    is to use 'continue' and let the iterator exhaust naturally.

    This test inspects the source code of the async delegate function to
    ensure no 'break' statement exists inside the streaming loop.
    """
    # Get the delegate functions — we need the async single-member variant
    # The function is a closure created inside get_delegate_functions, so
    # we'll verify the pattern via source inspection of _default_tools.py
    import agno.team._default_tools as module

    source = inspect.getsource(module)
    tree = ast.parse(source)

    # Find the adelegate_task_to_member function
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "adelegate_task_to_member":
            # Look for 'async for' loops in this function
            for child in ast.walk(node):
                if isinstance(child, ast.AsyncFor):
                    # Check that no Break statement exists inside this async for
                    for stmt in ast.walk(child):
                        assert not isinstance(stmt, ast.Break), (
                            "BUG #4688 regression: 'break' found inside 'async for' in "
                            "adelegate_task_to_member. This triggers GeneratorExit and causes "
                            "connection leaks. Use 'continue' instead."
                        )
            break
    else:
        raise AssertionError("Could not find adelegate_task_to_member in _default_tools.py")
