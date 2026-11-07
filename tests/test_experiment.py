import json
import re
from pathlib import Path

import torch

from experiments import evidence_heads
from experiments.build_states import build
from experiments.evidence_data import (
    bio_tags,
    bucket,
    build_prompt,
    decode_bio,
    field_evidence,
    load_rows,
    questions,
)


class FakeTokenizer:
    """Whitespace pieces with the leading space glued on, like Gemma's."""

    def __call__(self, text, return_offsets_mapping=True, add_special_tokens=True):
        offsets = [(m.start(), m.end()) for m in re.finditer(r"\s*\S+", text)]
        ids = [hash(text[a:b]) % 1000 for a, b in offsets]
        if add_special_tokens:
            ids = [2] + ids
            offsets = [(0, 0)] + offsets
        return {"input_ids": ids, "offset_mapping": offsets}


def fake_forward(ids, add_bos):
    generator = torch.Generator().manual_seed(len(ids))
    return torch.randn(2, len(ids), 8, generator=generator).half()


def _stored():
    rows = {}
    for path in ["data/audited/train.jsonl", "data/audited/val.jsonl", "data/audited/test.jsonl",
                 "data/round2/audited.jsonl", "data/round3/audited.jsonl", "data/generalization/locked.jsonl",
                 "data/exam2/locked.jsonl", "data/exam3/locked.jsonl", "data/exam4/locked.jsonl"]:
        for line in Path(path).read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["id"]] = row
    return rows


def test_reaudit_matches_stored_rows():
    rows = load_rows()
    stored = _stored()
    assert {row["id"] for row in rows} == set(stored)
    for row in rows:
        old = stored[row["id"]]
        assert row["text"] == old["text"]
        for name, field in row["fields"].items():
            assert field.get("value") == old["fields"][name].get("value")
    splits = {}
    for row in rows:
        splits[row["split"]] = splits.get(row["split"], 0) + 1
    assert splits == {"train": 208, "exam1": 24, "exam2": 24, "exam3": 24, "exam4": 24}


def test_every_scored_field_has_evidence_that_matches_the_text():
    for row in load_rows():
        for field in row["fields"].values():
            spans = field_evidence(field)
            if spans is None:
                assert field["type"] == "forecast"
                continue
            if field.get("missing"):
                assert spans == []
            else:
                assert spans
            for start, end in spans:
                assert 0 <= start < end <= len(row["text"])


def test_bio_roundtrip_and_buckets():
    tags = bio_tags(10, [(1, 2), (3, 3), (6, 8)])
    assert tags == [0, 1, 2, 1, 0, 0, 1, 2, 2, 0]
    assert decode_bio(tags) == [(1, 2), (3, 3), (6, 8)]
    assert decode_bio([2, 2, 0, 2]) == [(0, 1), (3, 3)]
    assert bucket([(1, 2)], [(1, 2)]) == "exact"
    assert bucket([], [(1, 2)]) == "missed"
    assert bucket([(5, 6)], []) == "false_alarm"
    assert bucket([(5, 6)], [(1, 2)]) == "wrong_place"
    assert bucket([(1, 1)], [(1, 2)]) == "partial"


def test_prompt_layouts_keep_document_offsets():
    text = "Fee 638.40 issued 4 June."
    full, start, spans = build_prompt("first", text, ["fee: Printed charge"])
    assert full[start:] == text
    assert full[spans[0][0]:spans[0][1]] == "fee: Printed charge"
    assert full.index("fee") < start
    full, start, spans = build_prompt("last", text, ["a: one", "b: two"])
    assert start == 0 and full.startswith(text)
    assert [full[a:b] for a, b in spans] == ["a: one", "b: two"]


def test_build_states_labels_land_on_the_marked_words():
    rows = [row for row in load_rows() if row["split"] == "exam4"][:3]
    tokenizer = FakeTokenizer()
    for prompt in ("last_nobos", "last", "first"):
        items, states, check = build(rows, tokenizer, fake_forward, prompt, log=lambda *_: None)
        assert len(items) == sum(len(questions(row)) for row in rows)
        assert check["forwards"] == (len(rows) if prompt != "first" else len(items))
        texts = {row["id"]: row["text"] for row in rows}
        for item in items:
            block = states[item["states_key"]]
            assert block.shape[1] == len(item["tags"])
            # Re-tokenize the bare note: the gold token positions must cover the gold words.
            offsets = [(m.start(), m.end()) for m in re.finditer(r"\s*\S+", texts[item["doc"]])]
            assert len(offsets) == block.shape[1]
            for first, last in item["gold"]:
                assert texts[item["doc"]][offsets[first][0]:offsets[last][1]].strip()


def test_tagger_learns_a_planted_signal():
    torch.manual_seed(0)
    states, items = {}, []
    for index in range(200):
        block = torch.randn(1, 20, 16)
        where = index % 15
        block[0, where, :4] += 4.0
        key = f"d{index}"
        states[key] = block.half()
        gold = [(where, where)]
        items.append({
            "states_key": key, "q_state": torch.randn(1, 16).half(), "end_state": torch.randn(1, 16).half(),
            "tags": torch.tensor(bio_tags(20, gold)), "gold": gold, "doc": key, "field": "f", "op": "copy",
            "type": "text", "split": "train", "value": None, "decimals": None,
        })
    model = None
    for _, model in evidence_heads.train_tagger(items[:160], states, "cpu", 20, (20,)):
        pass
    predicted = evidence_heads.predict(model, items[160:], states, "cpu")
    # Chance is about 0 of 40. The planted token should be found most of the time.
    assert evidence_heads.score(items[160:], predicted)["exact"] >= 28


def test_probe_runs_and_reports_a_baseline():
    torch.manual_seed(0)
    items = []
    for index in range(30):
        items.append({
            "doc": f"d{index}", "type": "number", "op": "read_number", "value": 100.0 + index, "decimals": 2,
            "split": "train", "end_state": torch.randn(2, 16).half(),
        })
    result = evidence_heads.probe(items, "end_state", "cpu", epochs=20)
    assert result["numbers"] == 30
    assert set(result["layers"]) == {0, 1}
    assert 0 <= result["baseline_exact"] <= 30


def test_pipeline_end_to_end_on_fake_states(monkeypatch):
    from experiments.report import summarize

    monkeypatch.setattr(evidence_heads, "EPOCHS", 2)
    monkeypatch.setattr(evidence_heads, "EVAL_EVERY", 1)
    rows = load_rows()
    picked = [r for r in rows if r["split"] == "train"][:15]
    for split in ("exam1", "exam2", "exam3"):
        picked += [r for r in rows if r["split"] == split][:2]
    items_by, states_by = {}, {}
    for prompt in ("last", "first"):
        items, states, _ = build(picked, FakeTokenizer(), fake_forward, prompt, log=lambda *_: None)
        items_by[prompt], states_by[prompt] = items, states
    results = evidence_heads.run_all(items_by, states_by, "cpu", log=lambda *_: None)
    payload = json.loads(json.dumps({"model": "fake", "layers": {"layers": [0, 1], "depth": 1},
                                     "seconds": 0, "results": results}, default=str))
    text = summarize(payload)
    assert "| last, dev exams 1-3 |" in text
    assert "| first, CV" in text
