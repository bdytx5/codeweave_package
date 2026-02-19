"""Demo: weave with local JSONL logging.

Run from real_weave/weave/:
    python demo_jsonl.py
"""

import json

import weave
from weave.trace_server_bindings.jsonl_logging_trace_server import attach_jsonl_logger


client = weave.init("my-demo-project")
log_path = attach_jsonl_logger(client)
print(f"Logging calls to: {log_path}\n")


# ---------------------------------------------------------------------------
# Define some ops
# ---------------------------------------------------------------------------

@weave.op
def add(a: int, b: int) -> int:
    return a + b


@weave.op
def greet(name: str, loud: bool = False) -> str:
    msg = f"Hello, {name}!"
    return msg.upper() if loud else msg


@weave.op
def divide(a: float, b: float) -> float:
    return a / b


# ---------------------------------------------------------------------------
# Call them
# ---------------------------------------------------------------------------

print("add(2, 3)                 =", add(2, 3))
print("add(10, -4)               =", add(10, -4))
print("greet('world')            =", greet("world"))
print("greet('world', loud=True) =", greet("world", loud=True))
print("divide(10, 4)             =", divide(10, 4))

try:
    divide(5, 0)
except ZeroDivisionError as e:
    print(f"divide(5, 0) raised:      {e}")

# Flush the batch processor so the log is written before we read it
weave.finish()

# ---------------------------------------------------------------------------
# Print the log
# ---------------------------------------------------------------------------

print("\n--- JSONL log ---")
with open(log_path) as f:
    for line in f:
        entry = json.loads(line)
        status = "ERROR" if entry["error"] else "OK"
        short_name = (entry["op_name"] or "").split("/")[-1].split(":")[0]
        dur = entry["duration_s"]
        dur_str = f"{dur:.4f}s" if dur is not None else "N/A"
        print(f"[{status}] {short_name}  inputs={entry['inputs']}  output={entry['output']}  duration={dur_str}")
