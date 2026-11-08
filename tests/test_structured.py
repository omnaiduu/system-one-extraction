import json

from experiments import structured_baseline as sb
from experiments.values import matches as _matches
from experiments.chains import test_variants as chain_test_variants
from experiments.evidence_data import load_rows
from experiments.make_kaggle_kernel import write_kernel


def _dev():
    return [r for r in load_rows() if r["split"] in ("exam1", "exam2", "exam3")]


def test_a_perfect_reply_scores_every_field():
    """The gold answers, as JSON, score 293/293: the scorer itself loses nothing."""
    items = []
    for row in _dev():
        fields = sb.field_list(row)
        reply = sb.parse_reply("Sure:\n" + sb.gold_json(fields))
        items += [(f, reply) for f in fields]
    result = sb.score_fields(items)
    assert result["fields"] == 293 and result["exact"] == 293


def test_lenient_reading_and_wrong_answers():
    spec = {"type": "number", "decimals": 2, "value": 1486.75, "missing": False}
    assert sb.to_prediction("Rs 1,486.75", spec)["value"] == 1486.75
    date_spec = {"type": "date", "value": "2026-06-04", "missing": False}
    assert sb.to_prediction("4 June 2026", date_spec)["value"] == "2026-06-04"
    assert sb.to_prediction(None, spec)["missing"]
    f = {"field": "fee", "type": "number", "spec": spec, "op": "read_number"}
    assert sb.score_fields([(f, {"fee": 1486.70})])["exact"] == 0
    assert sb.score_fields([(f, None)])["exact"] == 0  # an unparsable reply counts as wrong
    text = {"field": "name", "type": "text", "spec": {"type": "text", "value": "Adriatic Pipeworks d.o.o.", "missing": False}, "op": "copy"}
    scored = sb.score_fields([(text, {"name": "adriatic pipeworks doo"})])
    assert scored["exact"] == 0 and scored["exact_loose_text"] == 1


def test_prompts_schema_and_shots():
    rows = [r for r in load_rows() if r["split"] != "exam4"]
    shots = sb.pick_shots(rows)
    assert len(shots) == 3 and all(r["split"] == "train" for r, _ in shots)
    row = _dev()[0]
    fields = sb.field_list(row)
    messages = sb.messages_for(row, fields, shots)
    assert len(messages) == 8 and messages[0]["role"] == "system" and messages[-1]["role"] == "user"
    assert row["text"] in messages[-1]["content"] and json.loads(messages[2]["content"])  # gold JSON for the examples
    schema = sb.schema_for(fields)
    assert set(schema["properties"]) == {f["field"] for f in fields}
    computed = next(r for r in chain_test_variants(_dev()[:1], seed=23))
    only = sb.field_list(computed, only=["calc_0"])
    assert [f["field"] for f in only] == ["calc_0"]
    assert "calculate" in sb.SYSTEM and sb.SYSTEM.count("Answer:\n{") == 3


def test_structured_kernel_has_no_exam4_and_installs_the_enforcer(tmp_path):
    import base64, io, tarfile

    script = write_kernel(tmp_path / "s", "structured")
    text = script.read_text()
    assert "lm-format-enforcer" in text and "kaggle_structured_job import main" in text
    bundle = text.split("BUNDLE = ")[1].split("\n")[0].strip("'")
    names = tarfile.open(fileobj=io.BytesIO(base64.b64decode(bundle)), mode="r:gz").getnames()
    assert not any(n.startswith("data/exam4") for n in names)
    meta = json.loads((tmp_path / "s" / "kernel-metadata.json").read_text())
    assert meta["dataset_sources"] == ["urbikn/sroie-datasetv2"]


def test_system_prompt_examples_are_correct_and_not_from_the_exams():
    """The three worked answers parse as JSON, and their arithmetic is right."""
    answers = [json.loads(line) for line in sb.SYSTEM.splitlines() if line.startswith("{")]
    assert answers[0]["total"] == 1200 + 286.75 and answers[0]["due"] == "2026-06-25" and answers[0]["tracking"] is None
    assert answers[1]["final_price"] == round(2400 * 0.85 * 1.18, 2) and answers[1]["share"] == 900 / 4
    assert answers[1]["arrival"] == "2026-05-01" and answers[2]["amount"] == 120000 and answers[2]["nights"] == 4
    exam_text = " ".join(r["text"] for r in _dev())
    assert "Adriatic Pipeworks d.o.o. - invoice AP-2745" not in exam_text


def test_calculation_objects_reproduce_every_computed_gold_answer():
    """Inputs and steps built from the gold evidence, computed by code, give every gold answer."""
    from experiments.evidence_data import field_evidence, field_operation

    to_steps = {"sum": ["sum"], "subtract": ["subtract"], "product": ["multiply"], "add_days": ["add_days"],
                "days_between": ["days_between"], "median": ["median"]}
    checked = 0
    for row in [r for r in load_rows() if r["split"] != "exam4"]:
        for name, field in row["fields"].items():
            op = field_operation(field) if field_evidence(field) is not None else None
            if op not in to_steps:
                continue
            parts = [row["text"][a:b].strip() for a, b in field_evidence(field)]
            if op == "add_days":  # the date first, then the count, as the prompt asks
                parts.sort(key=lambda p: not any(ch.isalpha() for ch in p) and "-" not in p and "/" not in p)
            got = sb.to_prediction({"inputs": parts, "steps": to_steps[op]}, field)
            assert got["calculated"] and sb.score_fields(
                [({"field": name, "type": field["type"], "spec": field, "op": op}, {name: {"inputs": parts, "steps": to_steps[op]}})]
            )["exact"] == 1, (row["id"], name, parts, field["value"], got)
            checked += 1
    assert checked > 150
    for row in chain_test_variants(_dev(), seed=23):
        field = row["fields"]["calc_0"]
        if field["compute"] != "chain":
            continue
        values = field["span_values"]
        steps = field["blocks"]
        if field["chain_kind"] in ("percent_of", "discount") and float(values[0]) <= 100 and "." not in values[0]:
            values = [values[1], values[0]] + values[2:]  # amount first, rate next, as the prompt asks
        assert sb.to_prediction({"inputs": values, "steps": steps}, field)["value"] == field["value"], (field, values)


def test_malformed_calculations_are_wrong_not_crashes():
    spec = {"type": "number", "decimals": 2, "value": 10.0, "missing": False}
    for bad in ({"inputs": [1, 2], "steps": ["teleport"]}, {"inputs": [1], "steps": ["add"]}, {"inputs": "x", "steps": []},
                {"inputs": [1, 2, 3], "steps": ["add"]}):
        got = sb.to_prediction(bad, spec)
        assert got["value"] is None and not _matches(spec, got)
    answers = [json.loads(line) for line in sb.SYSTEM_CALC.splitlines() if line.startswith("{")]
    assert sb.evaluate_calculation(answers[1]["final_price"], spec) == 2407.2
    assert sb.evaluate_calculation(answers[2]["nights"], {"type": "duration"}) == 4


def test_hybrid_kernel_takes_the_model_and_receipts_but_not_exam4(tmp_path):
    script = write_kernel(tmp_path / "h", "hybrid", "ommnaidu/gemma-e2b-lora-onepass")
    meta = json.loads((tmp_path / "h" / "kernel-metadata.json").read_text())
    assert meta["kernel_sources"] == ["ommnaidu/gemma-e2b-lora-onepass"] and meta["dataset_sources"] == ["urbikn/sroie-datasetv2"]
    assert "kaggle_hybrid_job import main" in script.read_text()
