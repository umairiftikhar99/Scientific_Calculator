import ast
import math
import operator
import streamlit as st

# -----------------------------
# Safe math evaluator (AST-based)
# -----------------------------
_ALLOWED_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_ALLOWED_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

# Names allowed in expressions (functions/constants are injected later per angle mode)
_BASE_ENV = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
}

_BASE_FUNCS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "ln": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "pow": math.pow,
    "abs": abs,
    "floor": math.floor,
    "ceil": math.ceil,
    "factorial": math.factorial,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "asinh": math.asinh,
    "acosh": math.acosh,
    "atanh": math.atanh,
    "degrees": math.degrees,
    "radians": math.radians,
}

def _deg_wrap(func):
    return lambda x: func(math.radians(x))

def _deg_inv_wrap(invfunc):
    return lambda x: math.degrees(invfunc(x))

def make_env(angle_mode: str):
    env = dict(_BASE_ENV)
    env.update(_BASE_FUNCS)

    if angle_mode == "Degrees":
        env.update({
            "sin": _deg_wrap(math.sin),
            "cos": _deg_wrap(math.cos),
            "tan": _deg_wrap(math.tan),
            "asin": _deg_inv_wrap(math.asin),
            "acos": _deg_inv_wrap(math.acos),
            "atan": _deg_inv_wrap(math.atan),
        })
    else:
        env.update({
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
        })
    return env

class SafeEval(ast.NodeVisitor):
    def __init__(self, env):
        self.env = env

    def visit(self, node):
        if isinstance(node, ast.Expression):
            return self.visit(node.body)
        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("Invalid constant")
        elif isinstance(node, ast.BinOp):
            left = self.visit(node.left)
            right = self.visit(node.right)
            op_type = type(node.op)
            if op_type not in _ALLOWED_BINOPS:
                raise ValueError("Operator not allowed")
            if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right == 0:
                raise ZeroDivisionError("Division by zero")
            return _ALLOWED_BINOPS[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self.visit(node.operand)
            op_type = type(node.op)
            if op_type not in _ALLOWED_UNARYOPS:
                raise ValueError("Unary operator not allowed")
            return _ALLOWED_UNARYOPS[op_type](operand)
        elif isinstance(node, ast.Name):
            if node.id in self.env:
                return self.env[node.id]
            raise ValueError(f"Unknown identifier: {node.id}")
        elif isinstance(node, ast.Call):
            func = self.visit(node.func)
            args = [self.visit(a) for a in node.args]
            return func(*args)
        else:
            raise ValueError("Unsupported expression")

def safe_eval(expr: str, env):
    expr = expr.replace("^", "**")
    tree = ast.parse(expr, mode="eval")
    evaluator = SafeEval(env)
    return evaluator.visit(tree)

# -----------------------------
# UI helpers
# -----------------------------
def insert(tok: str):
    st.session_state.expr += tok

def backspace():
    st.session_state.expr = st.session_state.expr[:-1]

def clear():
    st.session_state.expr = ""

def _trim_trailing_identifier(text: str) -> str:
    """Remove a trailing function token (e.g. ``sin`` or ``log10``) from *text*.

    The helper makes sure we only drop identifiers that contain at least one
    alphabetic character so that we do not accidentally strip numeric values.
    """

    j = len(text) - 1
    while j >= 0 and text[j].isspace():
        j -= 1

    end = j
    while j >= 0 and (text[j].isalnum() or text[j] == '_'):
        j -= 1

    token = text[j + 1 : end + 1]
    if token and any(ch.isalpha() for ch in token):
        return text[: j + 1]
    return text[: end + 1]


def clear_entry():
    expr = st.session_state.expr.rstrip()
    if not expr:
        return

    i = len(expr) - 1
    while i >= 0 and expr[i].isspace():
        i -= 1

    if i < 0:
        st.session_state.expr = ""
        return

    if expr[i] == ')':
        depth = 1
        i -= 1
        while i >= 0 and depth > 0:
            if expr[i] == ')':
                depth += 1
            elif expr[i] == '(':
                depth -= 1
            i -= 1

        if depth > 0:
            # Unbalanced parentheses – fall back to clearing everything
            st.session_state.expr = ""
            return

        prefix = expr[: i + 1]
        st.session_state.expr = _trim_trailing_identifier(prefix).rstrip()
        return

    if expr[i].isalnum() or expr[i] == '.':
        while i >= 0 and (expr[i].isalnum() or expr[i] == '.'):
            i -= 1
    else:
        i -= 1

    st.session_state.expr = expr[: i + 1].rstrip()

def evaluate(angle_mode):
    expr = st.session_state.expr.strip()
    if not expr:
        return
    try:
        env = make_env(angle_mode)
        result = safe_eval(expr, env)
        st.session_state.history.insert(0, f"{expr} = {result}")
        st.session_state.expr = str(result)
    except ZeroDivisionError as zde:
        st.error(f"❌ {zde}")
    except Exception:
        st.error("❌ Invalid expression")

# -----------------------------
# App
# -----------------------------
st.set_page_config(page_title="Scientific Calculator", page_icon="🧮", layout="centered")

if "expr" not in st.session_state:
    st.session_state.expr = ""
if "history" not in st.session_state:
    st.session_state.history = []
if "angle_mode" not in st.session_state:
    st.session_state.angle_mode = "Degrees"

st.title("🧮 Scientific Calculator")

col_a, col_b = st.columns([2, 1])
with col_a:
    st.text_input("Expression", key="expr", placeholder="e.g., sin(30) + log(100, 10) * 3^2")
with col_b:
    st.session_state.angle_mode = st.radio("Angle", ["Degrees", "Radians"], horizontal=True)

# Keypad
rows = [
    ["(", ")", "⌫", "CE", "C"],
    ["sin(", "cos(", "tan(", "sqrt(", "log("],
    ["ln(", "log10(", "^", "÷", "×"],
    ["7", "8", "9", "-", "+"],
    ["4", "5", "6", ",", "."],
    ["1", "2", "3", "pi", "e"],
    ["0", "00", "tau", "=", ""],
]

token_map = {
    "×": "*", "÷": "/", "^": "^", "pi": "pi", "e": "e", "tau": "tau",
    ",": ","
}

for r in rows:
    cols = st.columns(len(r))
    for i, label in enumerate(r):
        if not label:
            continue

        def on_click_factory(lbl=label):
            def handler():
                if lbl == "=":
                    evaluate(st.session_state.angle_mode)
                elif lbl == "⌫":
                    backspace()
                elif lbl == "C":
                    clear()
                elif lbl == "CE":
                    clear_entry()
                else:
                    insert(token_map.get(lbl, lbl))
            return handler

        with cols[i]:
            st.button(label, on_click=on_click_factory())

if st.button("Evaluate"):
    evaluate(st.session_state.angle_mode)

if st.session_state.history:
    st.subheader("History")
    for item in st.session_state.history[:10]:
        st.code(item)
