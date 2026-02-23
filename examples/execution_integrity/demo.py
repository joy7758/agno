from execution_integrity_core import ExecutionIntegrityCore
from verify_export import verify_export

core = ExecutionIntegrityCore()

core.record("tool_call", {"tool": "search", "q": "agent execution"}, {"ok": True}, ts=1700000000.0)
core.record("tool_call", {"tool": "calc", "expr": "2+2"}, {"result": 4}, ts=1700000001.0)

print("Verification before tamper:", core.verify())

path = core.export(filename="execution_log.json", exported_at=1700000002.0)

core.chain[0]["output"] = {"ok": False}
print("Verification after tamper:", core.verify())

verify_export(path)
