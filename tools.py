"""Built-in tools the chat page can offer the model (OpenAI function-calling format).

The /v1 API itself stays spec-pure: clients send their own `tools` and execute the calls.
These are just the ones our own chat page uses, executed via POST /api/tools/run.
"""

import ast
import math
import operator
from datetime import datetime
from zoneinfo import ZoneInfo


def get_current_time(timezone="UTC"):
    try:
        now = datetime.now(ZoneInfo(timezone))
    except Exception:
        return f"Unknown timezone: {timezone}"
    return now.strftime(f"%Y-%m-%d %H:%M:%S ({timezone}, %A)")


_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}
_FUNCS = {name: getattr(math, name) for name in
          ["sqrt", "log", "log10", "log2", "exp", "sin", "cos", "tan", "floor", "ceil", "factorial"]}
_FUNCS.update(abs=abs, round=round, min=min, max=max)
_CONSTS = {"pi": math.pi, "e": math.e}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        if isinstance(node.op, ast.Pow) and abs(_eval(node.right)) > 1000:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    raise ValueError(f"unsupported expression: {ast.dump(node)[:60]}")


def calculate(expression):
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"Error: {e}"


TOOLS = {
    "get_current_time": (get_current_time, {
        "description": "Get the current date and time in a timezone.",
        "parameters": {"type": "object", "properties": {
            "timezone": {"type": "string", "description": "IANA timezone, e.g. 'Asia/Kolkata'. Defaults to UTC."},
        }},
    }),
    "calculate": (calculate, {
        "description": "Evaluate an arithmetic expression exactly. Supports + - * / // % ** and sqrt, log, exp, sin, cos, pi, e.",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "e.g. '17 * 23' or 'sqrt(2) * pi'"},
        }, "required": ["expression"]},
    }),
}


def schemas():
    return [{"type": "function", "function": {"name": name, **spec}} for name, (_, spec) in TOOLS.items()]


def run(name, arguments):
    if name not in TOOLS:
        return f"Error: unknown tool {name}"
    fn = TOOLS[name][0]
    try:
        return fn(**(arguments or {}))
    except TypeError as e:
        return f"Error: bad arguments: {e}"
