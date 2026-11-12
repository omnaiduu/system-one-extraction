import json
from datetime import date
from decimal import Decimal

import pytest

from experiments import formula as fm
from experiments import structured_baseline as sb
from experiments.augment import TEST_TEMPLATES, augment
from experiments.chains import test_variants as chain_test_variants
from experiments.evidence_data import field_evidence, field_operation, load_rows
from experiments.values import plain_number
from extractor.audit import Drop, parse_date


def _dev():
    return [r for r in load_rows() if r["split"] in ("exam1", "exam2", "exam3")]


NUM = {"type": "number", "decimals": 2}


def test_grammar():
    assert fm.answer("2400 * (1 - 15%) * (1 + 18%)", NUM) == 2407.2
    assert fm.answer("(4952.10 + 1993.42) / 7", NUM) == 992.22
    assert fm.answer("2026-10-09 + weeks(7)", {"type": "date"}) == "2026-11-27"
    assert fm.answer("date(2026-03-26) - days(25)", {"type": "date"}) == "2026-03-01"
    assert fm.answer('"2026-03-26" - days(25)', {"type": "date"}) == "2026-03-01"
    assert fm.answer("2026-03-07 - 2026-03-03", {"type": "duration"}) == 4
    assert fm.answer("days_between(2026-03-03, 2026-03-07)", {"type": "duration"}) == 4
    assert fm.answer("mean(80, 95, 120)", NUM) == 98.33
    assert fm.answer("median(80, 95, 120, 100)", {"type": "estimate", "decimals": 0}) == 98.0
    assert fm.answer("sum([1, 2.5])", NUM) == 3.5
    assert fm.answer("2026-04-10 + weeks(1) - days(2)", {"type": "date"}) == "2026-04-15"
    assert fm.answer("0.1 + 0.2", NUM) == 0.3  # Decimal, not float


@pytest.mark.parametrize("bad", ["__import__('os')", "2 ** 10", "x + 1", "2026-04-10 + 3", "days(2.5) + 2026-01-01",
                                 "open('f')", "1 / 0", "", "1" * 400, "2026-02-30 + days(1)", "(1).real", "[1, 2]",
                                 "mean()", "mod(5, 2)", "5 % 2"])
def test_outside_the_grammar_is_refused(bad):
    with pytest.raises(fm.FormulaError):
        fm.answer(bad, NUM)
    got = sb.to_prediction({"formula": bad}, {**NUM, "value": 1.0, "missing": False})
    assert got["value"] is None and got["calculated"]


def _read(part: str, kind: str):
    if kind == "date":
        return parse_date(part, True).isoformat()
    return str(plain_number(part))


def gold_formula(op: str, values: list[str], kinds: list[str]) -> str:
    """A formula a careful reader would write, from the gold evidence."""
    if op in ("sum", "average", "median"):
        return f"{ {'average': 'mean'}.get(op, op)}({', '.join(values)})"
    if op == "product":
        return " * ".join(values)
    if op == "subtract":
        return " - ".join(values)
    if op == "days_between":
        first, second = sorted(values)
        return f"{second} - {first}"
    if op == "add_days":
        d = next(v for v, k in zip(values, kinds) if k == "date")
        n = next(v for v, k in zip(values, kinds) if k != "date")
        return f"{d} + days({n})"
    raise ValueError(op)


def chain_formula(blocks: list[str], values: list[str], kinds: list[str]) -> str:
    def is_date(i):
        return kinds[i] == "date"

    if blocks[0] in ("percent_of", "minus_percent", "plus_percent") and float(values[0]) <= 100 and "." not in values[0]:
        values, kinds = [values[1], values[0]] + values[2:], [kinds[1], kinds[0]] + kinds[2:]
    if blocks[0].endswith(("days", "weeks")) and not is_date(0):
        values, kinds = [values[1], values[0]] + values[2:], [kinds[1], kinds[0]] + kinds[2:]
    acc = values[0]
    for block, x in zip(blocks, values[1:]):
        acc = {
            "add": f"({acc} + {x})", "subtract": f"({acc} - {x})", "multiply": f"({acc} * {x})", "divide": f"({acc} / {x})",
            "percent_of": f"({acc} * {x}%)", "minus_percent": f"({acc} * (1 - {x}%))", "plus_percent": f"({acc} * (1 + {x}%))",
            "add_days": f"({acc} + days({x}))", "minus_days": f"({acc} - days({x}))",
            "add_weeks": f"({acc} + weeks({x}))", "minus_weeks": f"({acc} - weeks({x}))",
        }[block]
    return acc


def _exam_formulas(rows):
    """(row, field name, spec, op, formula) for every calculated field whose evidence the gold marks."""
    out = []
    for row in rows:
        for name, field in row["fields"].items():
            op = field_operation(field) if field_evidence(field) is not None else None
            if op not in ("sum", "subtract", "product", "add_days", "days_between", "median", "average"):
                continue
            parts = [row["text"][a:b].strip() for a, b in field_evidence(field)]
            kinds = field.get("span_kinds") or [
                "date" if field["type"] == "duration" or (op == "add_days" and any(c.isalpha() for c in p) or "-" in p or "/" in p)
                and not p.isdigit() else "number" for p in parts]
            values = field.get("span_values") or [_read(p, k) for p, k in zip(parts, kinds)]
            out.append((row, name, field, op, gold_formula(op, values, kinds)))
    return out


def test_formulas_from_the_gold_reproduce_every_calculated_answer_and_pass_the_page_check():
    checked = 0
    for row, name, field, op, text in _exam_formulas([r for r in load_rows() if r["split"] != "exam4"]):
        assert fm.answer(text, field) == field["value"], (row["id"], name, text, field["value"])
        page = fm.page_values(row["text"], field.get("dayfirst", True))
        assert fm.check({"formula": text}, field, page) == [], (row["id"], name, text)
        checked += 1
    assert checked > 150
    for row in chain_test_variants(_dev(), seed=23):
        field = row["fields"]["calc_0"]
        if field["compute"] == "chain":
            text = chain_formula(field["blocks"], field["span_values"], field["span_kinds"])
        else:
            text = f"mean({', '.join(field['span_values'])})"
        assert fm.answer(text, field) == field["value"], (row["id"], text, field["value"])
        assert fm.check({"formula": text}, field, fm.page_values(row["text"])) == [], (row["id"], text)


def test_the_page_check_reads_numbers_and_dates_in_words():
    words_notes = augment(_dev(), seed=23, templates=TEST_TEMPLATES, emits=0)
    checked = 0
    for row in words_notes:
        page = fm.page_values(row["text"])
        for name, field in row["fields"].items():
            if field.get("missing") or field["type"] not in ("number", "date") or field_operation(field) not in ("read_number", "read_date"):
                continue
            value = field["value"]
            raw = value if field["type"] == "date" else float(value)
            assert fm.check(raw, field, page) == [], (row["id"], name, value)
            checked += 1
    assert checked > 50
    page = fm.page_values("Pay Rupees One Lakh Twenty Thousand Only, signed on the second of April, 2026, for four nights.")
    assert fm.number_on_page(Decimal(120000), page) and fm.number_on_page(Decimal(4), page)
    assert fm.date_on_page(date(2026, 4, 2), page) and not fm.date_on_page(date(2026, 4, 3), page)


def test_the_page_check_catches_invented_values():
    text = "Payment is due 7 weeks after 9 October 2026. Hotel 925.54 plus meals 416.55, with 20% service tax."
    page = fm.page_values(text)
    assert fm.check("2026-11-27", {"type": "date"}, page) == ["2026-11-27"]  # computed in its head
    assert fm.check({"formula": "2026-10-09 + weeks(7)"}, {"type": "date"}, page) == []
    assert fm.check({"formula": "(925.54 + 416.55) * (1 + 20%)"}, NUM, page) == []
    assert fm.check({"formula": "(925.54 + 416.55) * 1.2"}, NUM, page) == ["1.2"]
    assert fm.check(1610.51, NUM, page) == ["1610.51"]
    assert fm.check(925.54, NUM, page) == [] and fm.check(None, NUM, page) == [] and fm.check("Hotel", {"type": "text"}, page) == []


def test_formula_prompt_examples_are_right():
    answers = [json.loads(line) for line in sb.SYSTEM_FORMULA.splitlines() if line.startswith("{")]
    assert len(answers) == 3
    a, b, c = answers
    assert fm.answer(a["total"]["formula"], NUM) == 1486.75 and fm.answer(a["due"]["formula"], {"type": "date"}) == "2026-06-25"
    assert fm.answer(b["final_price"]["formula"], NUM) == 2407.2 and fm.answer(b["share"]["formula"], NUM) == 225
    assert fm.answer(b["arrival"]["formula"], {"type": "date"}) == "2026-05-01"
    assert fm.answer(c["nights"]["formula"], {"type": "duration"}) == 4
    assert fm.answer(c["typical"]["formula"], {"type": "estimate", "decimals": 0}) == 95
    # Every value in the example formulas is printed in its example document (the rule the prompt states).
    docs = [line for line in sb.SYSTEM_FORMULA.splitlines() if line.startswith(("Adriatic", "Subtotal", "Pay Rupees"))]
    for doc, ans in zip(docs, answers):
        page = fm.page_values(doc)
        for value in ans.values():
            assert fm.check(value, {"type": "date" if isinstance(value, str) else "number"}, page) == [], (doc, value)


# ---- the Kaggle job's logic, with a fake model ---------------------------------------

from experiments import kaggle_formula_job as job  # noqa: E402
from experiments.make_kaggle_kernel import write_kernel  # noqa: E402


def _calc_note(kind="combo_unseen"):
    return next(r for r in chain_test_variants(_dev(), seed=23) if r["chain_test"] == kind)


def test_answer_note_retries_only_fields_that_use_values_not_on_the_page():
    row = _calc_note()
    fields = sb.field_list(row, only=["calc_0"])
    field = fields[0]["spec"]
    good = chain_formula(field["blocks"], field["span_values"], field["span_kinds"])
    calls = []

    def fake(messages, asked):
        calls.append((messages, [f["field"] for f in asked]))
        if len(calls) == 1:
            return '{"calc_0": "2031-01-01"}', {"calc_0": "2031-01-01"}  # a date it made up
        return json.dumps({"calc_0": {"formula": good}}), {"calc_0": {"formula": good}}

    r = job.answer_note(row, fields, fake, sb.SYSTEM_FORMULA)
    assert r["flagged"] == ["calc_0"] and r["still"] == [] and r["retried"]
    assert "2031-01-01" in calls[1][0][-1]["content"] and calls[1][0][-2]["role"] == "assistant"
    scored = job.score_part([(fields, r)])
    assert scored["first"] == 0 and scored["after_retry"] == 1 and scored["accepted_right"] == 1 and scored["withheld"] == 0

    def honest(messages, asked):
        return "", {"calc_0": {"formula": good}}

    r = job.answer_note(row, fields, honest, sb.SYSTEM_FORMULA)
    assert not r["retried"] and job.score_part([(fields, r)])["first"] == 1

    def stubborn(messages, asked):
        return "", {"calc_0": "2031-01-01"}

    r = job.answer_note(row, fields, stubborn, sb.SYSTEM_FORMULA)
    s = job.score_part([(fields, r)])
    assert r["still"] == ["calc_0"] and s["withheld"] == 1 and s["accepted"] == 0


def test_schema_lets_numbers_and_dates_be_formulas():
    row = _dev()[0]
    fields = sb.field_list(row)
    schema = job.schema_with_formulas(fields)
    assert set(schema["required"]) == {f["field"] for f in fields} and schema["additionalProperties"] is False
    kinds = {f["field"]: f["type"] for f in fields}
    for name, prop in schema["properties"].items():
        has_formula = any(o.get("type") == "object" for o in prop["anyOf"])
        assert has_formula == (kinds[name] in ("number", "estimate", "duration", "date")), name
        assert {"type": "null"} in prop["anyOf"]


def test_formula_kernel_installs_xgrammar_reads_receipts_and_has_no_exam4(tmp_path):
    import base64, io, tarfile

    script = write_kernel(tmp_path / "f", "formula")
    text = script.read_text()
    assert '"xgrammar"' in text and "kaggle_formula_job import main" in text
    bundle = text.split("BUNDLE = ")[1].split("\n")[0].strip("'")
    names = tarfile.open(fileobj=io.BytesIO(base64.b64decode(bundle)), mode="r:gz").getnames()
    assert not any(n.startswith("data/exam4") for n in names) and "experiments/formula.py" in names
    meta = json.loads((tmp_path / "f" / "kernel-metadata.json").read_text())
    assert meta["dataset_sources"] == ["urbikn/sroie-datasetv2"] and meta["kernel_sources"] == []


def test_the_page_check_reads_receipt_dates():
    """The first formula run flagged correct receipt dates written 15-01-2019 or 15.01.19, and the retry broke them."""
    target = date(2019, 1, 15)
    for form in ["15/01/2019", "15-01-2019", "15.01.2019", "15/01/19", "15-01-19", "2019-01-15", "15 JAN 2019",
                 "15-JAN-2019", "15JAN2019", "Jan 15, 2019", "DATE:15/01/2019 11:30", "15.01.19 14:02"]:
        page = fm.page_values(f"TOTAL 12.50 {form} THANK YOU")
        assert fm.date_on_page(target, page), form
        assert fm.check("2019-01-15", {"type": "date"}, page) == [], form
    assert not fm.date_on_page(date(2019, 1, 16), fm.page_values("15-01-2019 TOTAL 12.50"))


def test_prompt_v2_only_adds_the_two_rules():
    assert sb.SYSTEM_FORMULA_V2.startswith(sb.SYSTEM_FORMULA.split("   - Follow the order")[0])
    added = set(sb.SYSTEM_FORMULA_V2.splitlines()) - set(sb.SYSTEM_FORMULA.splitlines())
    assert len(added) == 2 and all(line.startswith("   - ") for line in added)
    exam_text = " ".join(r["text"] for r in _dev())
    assert "2026-05-01" not in exam_text


def test_formula_v2_kernel(tmp_path):
    script = write_kernel(tmp_path / "v2", "formula_v2")
    assert "kaggle_formula_job import main_v2" in script.read_text() and '"xgrammar"' in script.read_text()


def test_head_to_head_kernel_carries_exam4_and_both_models(tmp_path):
    import base64, io, tarfile

    script = write_kernel(tmp_path / "hh", "head_to_head", "ommnaidu/gemma-e2b-lora-onepass,ommnaidu/gemma-e2b-lora-chains")
    text = script.read_text()
    assert "kaggle_head_to_head_job import main" in text and '"xgrammar"' in text
    bundle = text.split("BUNDLE = ")[1].split("\n")[0].strip("'")
    names = tarfile.open(fileobj=io.BytesIO(base64.b64decode(bundle)), mode="r:gz").getnames()
    assert any(n.startswith("data/exam4") for n in names)
    meta = json.loads((tmp_path / "hh" / "kernel-metadata.json").read_text())
    assert meta["kernel_sources"] == ["ommnaidu/gemma-e2b-lora-onepass", "ommnaidu/gemma-e2b-lora-chains"]
    assert meta["dataset_sources"] == []
    # Every other job still leaves exam 4 out.
    for job_name in ("formula", "formula_v2", "hybrid"):
        other = write_kernel(tmp_path / job_name, job_name, "ommnaidu/gemma-e2b-lora-onepass").read_text()
        b = other.split("BUNDLE = ")[1].split("\n")[0].strip("'")
        assert not any(n.startswith("data/exam4") for n in tarfile.open(fileobj=io.BytesIO(base64.b64decode(b)), mode="r:gz").getnames())


def test_head_to_head_summary_reads_both_systems():
    from experiments.kaggle_head_to_head_job import summarize

    part = {"first": 10, "after_retry": 12, "fields": 20, "accepted_right": 11, "accepted": 18}
    llm = {"modes": {"constrained": {"parts": {"exam4": part}}}}
    ours = {"exam4": {"values": {"exact": 9, "fields": 20}}}
    text = summarize({"notes": {"exam4": 24}, "llm": {"SYSTEM_FORMULA": llm, "SYSTEM_FORMULA_V2": llm},
                      "ours": {"experiment 4": ours, "experiment 6": {"error": "x"}}})
    assert "**12**/20" in text and "9/20" in text and "error" in text
