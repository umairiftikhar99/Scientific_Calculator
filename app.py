import ast
import csv
import io
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
STUDY_SUMMARY = {
    "study_design": {
        "n": 225,
        "likert_scale": [1, 5],
        "constructs": {
            "AI": {"items": 5, "mean": 3.79, "sd": 0.65, "alpha": 0.84},
            "TeamCollaboration": {"items": 5, "mean": 3.58, "sd": 0.72, "alpha": 0.88},
            "ProjectComplexity": {"items": 5, "mean": 3.19, "sd": 0.69, "alpha": 0.81},
            "ProjectSuccess": {"items": 6, "mean": 3.89, "sd": 0.71, "alpha": 0.91},
        },
    },
    "correlations": {
        "AI_TeamCollaboration": 0.83,
        "AI_ProjectComplexity": -0.12,
        "AI_ProjectSuccess": 0.34,
        "TeamCollaboration_ProjectComplexity": -0.15,
        "TeamCollaboration_ProjectSuccess": 0.42,
        "ProjectComplexity_ProjectSuccess": -0.25,
    },
    "multiple_regression": {
        "DV": "ProjectSuccess",
        "predictors": [
            {"name": "AI", "B": 0.21, "SE": 0.08, "Beta": 0.21, "t": 2.63, "p": 0.009},
            {
                "name": "TeamCollaboration",
                "B": 0.36,
                "SE": 0.07,
                "Beta": 0.36,
                "t": 5.14,
                "p": 0.000,
            },
            {
                "name": "ProjectComplexity",
                "B": -0.18,
                "SE": 0.07,
                "Beta": -0.18,
                "t": -2.57,
                "p": 0.011,
            },
        ],
        "constant": {"B": 2.11, "SE": 0.32, "t": 6.59, "p": 0.000},
        "model_fit": {"R2": 0.239, "AdjR2": 0.229, "F_df": [3, 221], "F": 23.11, "p": "< .001"},
    },
    "mediation_MODEL4": {
        "X": "AI",
        "M": "TeamCollaboration",
        "Y": "ProjectSuccess",
        "PROCESS_settings": {"bootstrap": 5000, "ci_level": 0.95},
        "paths": {
            "a_AI_to_M": {"B_or_Beta": 1.03, "SE": 0.08, "t": 12.88, "p": 0.000},
            "b_M_to_Y": {"B_or_Beta": 0.36, "SE": 0.07, "t": 5.14, "p": 0.000},
            "c_total_AI_to_Y": {"B_or_Beta": 0.70, "SE": 0.09, "t": 7.78, "p": 0.000},
            "c_prime_direct_AI_to_Y": {"B_or_Beta": 0.09, "SE": 0.08, "t": 1.13, "p": 0.261},
        },
        "conclusion": "Full mediation (direct effect non-significant when M included)",
    },
    "moderation": {
        "model": "AI * ProjectComplexity → ProjectSuccess",
        "effects": {
            "AI_main": {"B": 0.22, "SE": 0.09, "t": 2.44, "p": 0.015},
            "Complexity_main": {"B": -0.17, "SE": 0.07, "t": -2.43, "p": 0.016},
            "AIxComplexity_interaction": {"B": 0.52, "SE": 0.46, "t": 1.12, "p": 0.264},
        },
        "conclusion": "No significant moderation",
    },
}


def _markdown_table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _rows_to_csv(headers, rows):
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(headers)
    writer.writerows(rows)
    return buffer.getvalue()


def render_calculator():
    st.title("🧮 Scientific Calculator")

    col_a, col_b = st.columns([2, 1])
    with col_a:
        st.text_input(
            "Expression",
            key="expr",
            placeholder="e.g., sin(30) + log(100, 10) * 3^2",
        )
    with col_b:
        st.session_state.angle_mode = st.radio(
            "Angle", ["Degrees", "Radians"], horizontal=True
        )

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
        "×": "*",
        "÷": "/",
        "^": "^",
        "pi": "pi",
        "e": "e",
        "tau": "tau",
        ",": ",",
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


def _render_constructs(constructs):
    headers = ["Construct", "Items", "Mean", "SD", "α"]
    csv_rows = []
    display_rows = []

    for name, stats in constructs.items():
        csv_rows.append(
            [name, stats["items"], stats["mean"], stats["sd"], stats["alpha"]]
        )
        display_rows.append(
            [
                name,
                stats["items"],
                f"{stats['mean']:.2f}",
                f"{stats['sd']:.2f}",
                f"{stats['alpha']:.2f}",
            ]
        )

    st.markdown(_markdown_table(headers, display_rows))
    st.download_button(
        "Download construct descriptives (CSV)",
        data=_rows_to_csv(headers, csv_rows),
        file_name="constructs.csv",
        mime="text/csv",
    )


def _render_correlations(correlations):
    headers = ["Pair", "r"]
    display_rows = []
    csv_rows = []

    for key, value in correlations.items():
        nice_key = key.replace("_", " ↔ ")
        display_rows.append([nice_key, f"{value:+.2f}"])
        csv_rows.append([nice_key, value])

    st.markdown(_markdown_table(headers, display_rows))
    st.download_button(
        "Download correlations (CSV)",
        data=_rows_to_csv(headers, csv_rows),
        file_name="correlations.csv",
        mime="text/csv",
    )


def _render_regression(regression):
    headers = ["Predictor", "B", "SE", "β", "t", "p"]
    display_rows = []
    csv_rows = []

    for pred in regression["predictors"]:
        display_rows.append(
            [
                pred["name"],
                f"{pred['B']:.2f}",
                f"{pred['SE']:.2f}",
                f"{pred['Beta']:.2f}",
                f"{pred['t']:.2f}",
                f"{pred['p']:.3f}",
            ]
        )
        csv_rows.append(
            [
                pred["name"],
                pred["B"],
                pred["SE"],
                pred["Beta"],
                pred["t"],
                pred["p"],
            ]
        )

    constant = regression.get("constant")
    if constant:
        display_rows.append(
            [
                "Constant",
                f"{constant['B']:.2f}",
                f"{constant['SE']:.2f}",
                "—",
                f"{constant['t']:.2f}",
                f"{constant['p']:.3f}",
            ]
        )
        csv_rows.append(
            [
                "Constant",
                constant["B"],
                constant["SE"],
                None,
                constant["t"],
                constant["p"],
            ]
        )

    st.markdown(_markdown_table(headers, display_rows))
    st.download_button(
        "Download regression coefficients (CSV)",
        data=_rows_to_csv(headers, csv_rows),
        file_name="regression_coefficients.csv",
        mime="text/csv",
    )

    fit = regression["model_fit"]
    st.info(
        "Model fit: "
        f"R² = {fit['R2']:.3f}, Adjusted R² = {fit['AdjR2']:.3f}, "
        f"F({fit['F_df'][0]}, {fit['F_df'][1]}) = {fit['F']:.2f}, p {fit['p']}"
    )

    fit_headers = ["Metric", "Value"]
    fit_rows = [
        ["R2", fit["R2"]],
        ["Adjusted R2", fit["AdjR2"]],
        ["F_df", ", ".join(map(str, fit["F_df"]))],
        ["F", fit["F"]],
        ["p", fit["p"]],
    ]
    st.download_button(
        "Download regression model fit (CSV)",
        data=_rows_to_csv(fit_headers, fit_rows),
        file_name="regression_model_fit.csv",
        mime="text/csv",
    )


def render_study_insights():
    st.title("📊 Study Insights")

    design = STUDY_SUMMARY["study_design"]
    st.subheader("Study Design")
    st.markdown(
        f"- Sample size: **n = {design['n']}**\n"
        f"- Likert scale range: **{design['likert_scale'][0]}–{design['likert_scale'][1]}**"
    )

    st.subheader("Construct Reliability & Descriptives")
    _render_constructs(design["constructs"])

    st.subheader("Bivariate Correlations")
    _render_correlations(STUDY_SUMMARY["correlations"])

    st.subheader("Multiple Regression (DV: Project Success)")
    _render_regression(STUDY_SUMMARY["multiple_regression"])

    mediation = STUDY_SUMMARY["mediation_MODEL4"]
    st.subheader("Mediation (PROCESS Model 4)")
    mediation_headers = ["Path", "Relationship", "B/Beta", "SE", "t", "p"]
    mediation_rows = [
        [
            "a",
            f"{mediation['X']} → {mediation['M']}",
            mediation["paths"]["a_AI_to_M"]["B_or_Beta"],
            mediation["paths"]["a_AI_to_M"]["SE"],
            mediation["paths"]["a_AI_to_M"]["t"],
            mediation["paths"]["a_AI_to_M"]["p"],
        ],
        [
            "b",
            f"{mediation['M']} → {mediation['Y']}",
            mediation["paths"]["b_M_to_Y"]["B_or_Beta"],
            mediation["paths"]["b_M_to_Y"]["SE"],
            mediation["paths"]["b_M_to_Y"]["t"],
            mediation["paths"]["b_M_to_Y"]["p"],
        ],
        [
            "c",
            f"{mediation['X']} → {mediation['Y']}",
            mediation["paths"]["c_total_AI_to_Y"]["B_or_Beta"],
            mediation["paths"]["c_total_AI_to_Y"]["SE"],
            mediation["paths"]["c_total_AI_to_Y"]["t"],
            mediation["paths"]["c_total_AI_to_Y"]["p"],
        ],
        [
            "c'",
            f"{mediation['X']} → {mediation['Y']} | {mediation['M']}",
            mediation["paths"]["c_prime_direct_AI_to_Y"]["B_or_Beta"],
            mediation["paths"]["c_prime_direct_AI_to_Y"]["SE"],
            mediation["paths"]["c_prime_direct_AI_to_Y"]["t"],
            mediation["paths"]["c_prime_direct_AI_to_Y"]["p"],
        ],
    ]
    mediation_display_rows = [
        [label, rel, f"{coef:.2f}", f"{se:.2f}", f"{tval:.2f}", "< .001" if p == 0 else f"{p:.3f}"]
        for label, rel, coef, se, tval, p in mediation_rows
    ]
    st.markdown(_markdown_table(mediation_headers, mediation_display_rows))
    st.download_button(
        "Download mediation paths (CSV)",
        data=_rows_to_csv(mediation_headers, mediation_rows),
        file_name="mediation_paths.csv",
        mime="text/csv",
    )
    st.success(mediation["conclusion"])

    moderation = STUDY_SUMMARY["moderation"]
    st.subheader("Moderation: AI × Project Complexity")
    moderation_headers = ["Effect", "B", "SE", "t", "p"]
    moderation_rows = []
    for name, stats in moderation["effects"].items():
        moderation_rows.append(
            [
                name,
                stats["B"],
                stats["SE"],
                stats["t"],
                stats["p"],
            ]
        )

    moderation_display_rows = [
        [row[0], f"{row[1]:.2f}", f"{row[2]:.2f}", f"{row[3]:.2f}", f"{row[4]:.3f}"]
        for row in moderation_rows
    ]
    st.markdown(_markdown_table(moderation_headers, moderation_display_rows))
    st.download_button(
        "Download moderation effects (CSV)",
        data=_rows_to_csv(moderation_headers, moderation_rows),
        file_name="moderation_effects.csv",
        mime="text/csv",
    )
    st.warning(moderation["conclusion"])

    st.subheader("Key Takeaways")
    st.markdown(
        "- **Team collaboration is a strong lever** for project success, both directly and as a mediator of the AI effect.\n"
        "- **AI support boosts outcomes primarily through better collaboration**, consistent with the full mediation finding.\n"
        "- **Higher project complexity hinders success**, yet the AI × complexity interaction was not significant, suggesting AI benefits are stable across complexity levels."
    )


st.set_page_config(page_title="Scientific Calculator", page_icon="🧮", layout="centered")

if "expr" not in st.session_state:
    st.session_state.expr = ""
if "history" not in st.session_state:
    st.session_state.history = []
if "angle_mode" not in st.session_state:
    st.session_state.angle_mode = "Degrees"

mode = st.sidebar.radio("Mode", ["Calculator", "Study insights"], index=0)

if mode == "Calculator":
    render_calculator()
else:
    render_study_insights()
