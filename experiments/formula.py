"""The LLM writes a calculation as an ordinary formula; code evaluates it exactly and checks it.

Why: with the step vocabulary of note 18 ({"inputs", "steps"}) most wrong
calculations were translation errors into our own names: "7 weeks after"
written as add_days 7, "25 days ahead of" as add_days, "(hotel + meals) plus
20% tax on everything" with invented steps, a mean as ["sum", "divide"] with
the count missing. A formula is what the model has seen millions of times,
and it can say the order with brackets:

    2400 * (1 - 15%) * (1 + 18%)          -> 2407.20
    (4952.10 + 1993.42) / 7               -> 992.22
    2026-10-09 + weeks(7)                 -> 2026-11-27
    2026-03-26 - days(25)                 -> 2026-03-01
    mean(80, 95, 120)   median(...)   sum(...)
    2026-03-07 - 2026-03-03               -> 4 (days)

Allowed: numbers, ISO dates (bare, quoted, or in date(...)), + - * /,
brackets, n% (n/100), days(n), weeks(n), sum, mean/average, median, min, max,
days_between. Parsed with `ast` and walked node by node, never `eval`. All
arithmetic in Decimal; money is rounded once, at the end.

`on_page` is the reliability check: every number and date the model used (or
gave as a literal answer) must be printed in the document, as digits or in
words. A value that is not on the page was invented or computed in the
model's head.
"""

from __future__ import annotations

import ast
import re
from datetime import date, timedelta
from decimal import Decimal, InvalidOperation, localcontext

from experiments.rewrite import date_forms, number_forms, words
from experiments.values import _DATES, _NUMBER
from extractor.audit import Drop, parse_date, quantize

MAX_LENGTH = 300
CONSTANTS = {Decimal(0), Decimal(1), Decimal(100)}  # allowed in a formula without being on the page
_ISO = re.compile(r"""date\(\s*["']?(\d{4}-\d{2}-\d{2})["']?\s*\)|["']?(\d{4}-\d{2}-\d{2})["']?""")
_PERCENT = re.compile(r"(\d+(?:\.\d+)?)\s*%")


class FormulaError(ValueError):
    pass


def _prepare(text: str) -> tuple[str, list[date], list[str]]:
    """Dates become names (__d0, __d1, ...), n% becomes (n/100). Returns source, dates, rate literals."""
    if not isinstance(text, str) or not text.strip() or len(text) > MAX_LENGTH:
        raise FormulaError("empty or too long")
    source = text.strip().replace("×", "*").replace("÷", "/").replace("−", "-").replace("–", "-")
    dates: list[date] = []

    def name_date(match):
        iso = match.group(1) or match.group(2)
        try:
            dates.append(date.fromisoformat(iso))
        except ValueError as exc:
            raise FormulaError(f"bad date {iso}") from exc
        return f"__d{len(dates) - 1}"

    source = _ISO.sub(name_date, source)
    rates = _PERCENT.findall(source)
    source = _PERCENT.sub(r"(\1/100)", source)
    if "%" in source:
        raise FormulaError("% after something that is not a number")
    return source, dates, rates


def _days(value) -> int:
    if not isinstance(value, Decimal) or value != value.to_integral_value():
        raise FormulaError("days(...) and weeks(...) need a whole number")
    return int(value)


def _numbers(args) -> list[Decimal]:
    if len(args) == 1 and isinstance(args[0], list):
        args = args[0]
    if not args or not all(isinstance(a, Decimal) for a in args):
        raise FormulaError("needs numbers")
    return list(args)


def _median(args):
    values = sorted(_numbers(args))
    mid = len(values) // 2
    return values[mid] if len(values) % 2 else (values[mid - 1] + values[mid]) / 2


def _days_between(args):
    if len(args) != 2 or not all(isinstance(a, date) for a in args):
        raise FormulaError("days_between needs two dates")
    return Decimal(abs((args[1] - args[0]).days))


FUNCTIONS = {
    "days": lambda args: timedelta(days=_days(args[0])) if len(args) == 1 else _bad("days"),
    "weeks": lambda args: timedelta(weeks=_days(args[0])) if len(args) == 1 else _bad("weeks"),
    "sum": lambda args: sum(_numbers(args), Decimal(0)),
    "mean": lambda args: sum(_numbers(args), Decimal(0)) / len(_numbers(args)),
    "median": _median,
    "min": lambda args: min(_numbers(args)),
    "max": lambda args: max(_numbers(args)),
    "days_between": _days_between,
}
FUNCTIONS["average"] = FUNCTIONS["avg"] = FUNCTIONS["mean"]


def _bad(name):
    raise FormulaError(f"{name}(...) takes one argument")


def _binary(op, a, b):
    if isinstance(op, ast.Add):
        if isinstance(a, Decimal) and isinstance(b, Decimal):
            return a + b
        if isinstance(a, date) and isinstance(b, timedelta):
            return a + b
        if isinstance(a, timedelta) and isinstance(b, date):
            return b + a
        if isinstance(a, timedelta) and isinstance(b, timedelta):
            return a + b
    elif isinstance(op, ast.Sub):
        if isinstance(a, Decimal) and isinstance(b, Decimal):
            return a - b
        if isinstance(a, date) and isinstance(b, timedelta):
            return a - b
        if isinstance(a, date) and isinstance(b, date):
            return Decimal((a - b).days)
        if isinstance(a, timedelta) and isinstance(b, timedelta):
            return a - b
    elif isinstance(op, ast.Mult):
        if isinstance(a, Decimal) and isinstance(b, Decimal):
            return a * b
    elif isinstance(op, ast.Div):
        if isinstance(a, Decimal) and isinstance(b, Decimal):
            if b == 0:
                raise FormulaError("division by zero")
            return a / b
    raise FormulaError(f"cannot {type(op).__name__} {type(a).__name__} and {type(b).__name__}")


def _walk(node, source: str, dates: list[date]):
    if isinstance(node, ast.Expression):
        return _walk(node.body, source, dates)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        try:
            return Decimal(ast.get_source_segment(source, node))
        except (InvalidOperation, TypeError) as exc:
            raise FormulaError("bad number") from exc
    if isinstance(node, ast.Name) and re.fullmatch(r"__d\d+", node.id):
        return dates[int(node.id[3:])]
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _walk(node.operand, source, dates)
        if not isinstance(value, Decimal):
            raise FormulaError("sign on a date")
        return -value if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
        return _binary(node.op, _walk(node.left, source, dates), _walk(node.right, source, dates))
    if isinstance(node, ast.List):
        return [_walk(e, source, dates) for e in node.elts]
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id.lower() in FUNCTIONS and not node.keywords:
        return FUNCTIONS[node.func.id.lower()]([_walk(a, source, dates) for a in node.args])
    raise FormulaError(f"not allowed: {ast.dump(node)[:60]}")


def evaluate(text: str):
    """Decimal, date or timedelta. Raises FormulaError on anything outside the grammar."""
    source, dates, _ = _prepare(text)
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise FormulaError("not a formula") from exc
    with localcontext() as ctx:
        ctx.prec = 34
        try:
            return _walk(tree, source, dates)
        except (ArithmeticError, IndexError, OverflowError) as exc:
            raise FormulaError(str(exc)) from exc


def answer(text: str, spec: dict):
    """The formula's value in the field's type (ISO date, whole days, or money rounded at the end)."""
    value = evaluate(text)
    kind = spec["type"]
    if kind == "date":
        if not isinstance(value, date):
            raise FormulaError("a date field needs a date")
        return value.isoformat()
    if isinstance(value, timedelta):
        value = Decimal(value.days)
    if isinstance(value, date) or isinstance(value, list):
        raise FormulaError("a number field needs a number")
    if kind == "duration":
        return int(quantize(value, 0))
    outer = re.match(r"\s*(\w+)\s*\(", text)
    whole = outer and outer.group(1).lower() == "median" and text.rstrip().endswith(")")
    decimals = (spec.get("decimals") or 0) if whole else int(spec.get("decimals", 2))
    return float(quantize(value, int(decimals)))


def inputs(text: str) -> tuple[list[Decimal], list[date]]:
    """The numbers and dates a formula uses (rates as written, e.g. 15 for 15%)."""
    source, dates, _ = _prepare(text)
    try:
        tree = ast.parse(source, mode="eval")
    except SyntaxError as exc:
        raise FormulaError("not a formula") from exc
    numbers = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            numbers.append(Decimal(ast.get_source_segment(source, node)))
    return numbers, dates


# ---- the page check --------------------------------------------------------------


def _windows(tokens: list[str], size: int):
    for i in range(len(tokens) - size + 1):
        yield tokens[i : i + size]


def page_values(text: str, dayfirst: bool = True) -> tuple[set[Decimal], set[date], list[str]]:
    """Every number and date printed as digits, and the word tokens for numbers and dates in words."""
    numbers = set()
    for match in _NUMBER.finditer(text):
        try:
            numbers.add(Decimal(match.group(0).replace(",", "")))
        except InvalidOperation:
            pass
    dates = set()
    for pattern in _DATES:
        for match in pattern.finditer(text):
            for first in (dayfirst, not dayfirst):
                try:
                    dates.add(parse_date(match.group(0), first))
                except Drop:
                    pass
    return numbers, dates, words(text)


def number_on_page(value: Decimal, page) -> bool:
    numbers, _, tokens = page
    if value in numbers:
        return True
    for form in number_forms(value):
        if any(window == form for window in _windows(tokens, len(form))):
            return True
    return False


def date_on_page(value: date, page) -> bool:
    _, dates, tokens = page
    if value in dates:
        return True
    for form in date_forms(value):
        target = sorted(form)
        if any(sorted(window) == target for window in _windows(tokens, len(form))):
            return True
    return False


def check(raw, spec: dict, page) -> list[str]:
    """Values in this answer that are not printed in the document (empty list: the answer passes).

    A formula: every number (except 0, 1, 100) and date it uses. A literal
    number, number of days or date: the value itself. Text, lists, times and
    nulls are not checked here.
    """
    if raw is None:
        return []
    if isinstance(raw, dict) and "formula" in raw:
        try:
            numbers, dates = inputs(str(raw["formula"]))
        except FormulaError:
            return []  # unparsable: wrong anyway, nothing to check
        bad = [str(n) for n in numbers if n not in CONSTANTS and not number_on_page(n, page)]
        return bad + [d.isoformat() for d in dates if not date_on_page(d, page)]
    kind = spec["type"]
    if kind in ("number", "estimate", "duration") and isinstance(raw, (int, float)) and not isinstance(raw, bool):
        value = Decimal(str(raw))
        return [] if number_on_page(value, page) else [str(raw)]
    if kind == "date" and isinstance(raw, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", raw.strip()):
        try:
            value = date.fromisoformat(raw.strip())
        except ValueError:
            return []
        return [] if date_on_page(value, page) else [raw]
    return []
