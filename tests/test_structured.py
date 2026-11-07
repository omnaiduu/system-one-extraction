import json

from experiments import structured_baseline as sb
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
    assert len(messages) == 7 and messages[-1]["role"] == "user" and row["text"] in messages[-1]["content"]
    assert json.loads(messages[1]["content"])  # gold JSON for the examples
    schema = sb.schema_for(fields)
    assert set(schema["properties"]) == {f["field"] for f in fields}
    computed = next(r for r in chain_test_variants(_dev()[:1], seed=23))
    only = sb.field_list(computed, only=["calc_0"])
    assert [f["field"] for f in only] == ["calc_0"]
    assert "calculate" in sb.RULES


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
