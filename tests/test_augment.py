import random
from datetime import date
from decimal import Decimal

import pytest
import torch

from experiments import augment, lora_tagger
from experiments.evidence_data import field_evidence, field_operation, load_rows, questions
from experiments.values import ALLOWED, answer, matches
from extractor.audit import Drop, parse_date, parse_number


@pytest.fixture(scope="module")
def rows():
    return [r for r in load_rows() if r["split"] != "exam4"]


def _variants(rows):
    train = [r for r in rows if r["split"] == "train"]
    dev = [r for r in rows if r["split"] in ("exam1", "exam2", "exam3")]
    return augment.augment(train, seed=11) + augment.augment(dev, seed=23, templates=augment.TEST_TEMPLATES)


def test_number_and_date_words():
    rng = random.Random(0)
    assert augment.integer_words(1486, random.Random(1)).replace(" and", "") == "one thousand four hundred eighty-six"
    assert augment.integer_words(250000, random.Random(3)) in ("two hundred fifty thousand", "two hundred and fifty thousand", "two lakh fifty thousand")
    for value in ("638.40", "1486.75", "12.00", "45"):
        words = augment.number_words(Decimal(value), 2, rng)
        with pytest.raises(Drop):
            parse_number(words)  # the parser cannot read words: the learned reader must
    words = augment.date_words(date(2026, 6, 4), rng)
    assert "June" in words
    with pytest.raises(Drop):
        parse_date(words)


def test_every_variant_is_consistent_with_a_perfect_reader(rows):
    """Gold spans + a reader that returns each span's true value + the true emit value = gold answer."""
    checked = 0
    for row in _variants(rows):
        for name, field in row["fields"].items():
            spans = field_evidence(field)
            if spans is None:
                continue
            op = field_operation(field)
            assert op == "missing" or op in ALLOWED[field["type"]], (row["id"], name, op)
            parts = [row["text"][a:b].strip() for a, b in spans]
            targets = lora_tagger.span_targets(row, field)

            def read(index, kind):
                target = targets[index]
                if target is None:
                    return None
                if kind == "date":
                    return target[1] if isinstance(target[1], date) else None
                return int(target[1]) if kind == "count" else target[1]

            emitted = field["value"] if op == "emit" else None
            predicted = answer(ALLOWED[field["type"]][0] if op == "missing" else op, parts, field, read=read, emitted=emitted)
            assert matches(field, predicted), (row["id"], name, parts, field.get("value"), predicted)
            checked += 1
    assert checked > 2000


def test_words_variants_keep_spans_on_the_words(rows):
    variants = [v for v in _variants(rows) if v["augment"] == "words"]
    assert len(variants) > 250
    rewritten = 0
    for row in variants:
        for field in row["fields"].values():
            for a, b in field_evidence(field) or []:
                assert 0 <= a < b <= len(row["text"])
                text = row["text"][a:b]
                assert text == text.strip() or text[0] == " "
            if field.get("type") == "text" and not field.get("missing"):
                assert row["text"][field["start"]:field["end"]] == field["value"]
        for field in row["fields"].values():
            if field.get("type") in ("number", "date", "duration"):
                for a, b in field_evidence(field) or []:
                    rewritten += any(ch.isalpha() for ch in row["text"][a:b]) and any(
                        w in row["text"][a:b] for w in ("hundred", "thousand", " of ", " the ", "teen", "ty"))
    assert rewritten > 300


def test_emit_variants_compute_and_hide_the_answer(rows):
    variants = [v for v in _variants(rows) if v["augment"] == "emit"]
    assert len(variants) > 250
    for row in variants:
        name = [k for k in row["fields"] if k.startswith("emit_")][0]
        field = row["fields"][name]
        assert field["compute"] == "emit"
        assert [row["text"][e["start"]:e["end"]] for e in field["evidence"]] == [e["text"] for e in field["evidence"]]
        if field["type"] == "number":
            assert f"{field['value']:,.2f}" not in row["text"]
        else:
            assert field["value"] not in row["text"]


def test_exam_phrasings_are_never_used_in_training():
    train_sentences = {s for _, s, _ in augment.TRAIN_TEMPLATES}
    train_instructions = {i for _, _, i in augment.TRAIN_TEMPLATES}
    for _, sentence, instruction in augment.TEST_TEMPLATES:
        assert sentence not in train_sentences and instruction not in train_instructions


def test_variants_share_their_source_note_fold(rows):
    from tests.test_lora import FakeTokenizer

    train = [r for r in rows if r["split"] == "train"][:30]
    variants = augment.augment(train, seed=11)
    parts = lora_tagger.assign_parts(lora_tagger.encode(train + variants, FakeTokenizer(), "multi"))
    held = {augment.base_id(ex["doc"]) for ex in parts["held"]}
    fit = {augment.base_id(ex["doc"]) for ex in parts["fit"]}
    assert held and fit and not held & fit


def test_reader_learns_a_planted_number_and_date():
    torch.manual_seed(0)
    reader = lora_tagger.SpanReader(1, 16, lora_tagger.NUMBER_OUTPUTS, inner=32)
    tokens = torch.randn(1, 3, 16)
    question = torch.randn(1, 16)
    target = lora_tagger.number_target(Decimal("638.40"))
    opt = torch.optim.Adam(reader.parameters(), lr=1e-2)
    for _ in range(150):
        loss = lora_tagger.reader_loss(reader, tokens, question, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    reader.eval()
    assert lora_tagger.read_number(reader, tokens, question) == Decimal("638.40")
    dreader = lora_tagger.SpanReader(1, 16, lora_tagger.DATE_OUTPUTS, inner=32)
    target = lora_tagger.date_target(date(2026, 6, 4))
    opt = torch.optim.Adam(dreader.parameters(), lr=1e-2)
    for _ in range(150):
        loss = lora_tagger.reader_loss(dreader, tokens, question, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    dreader.eval()
    assert lora_tagger.read_date(dreader, tokens, question) == date(2026, 6, 4)


def test_emit_training_runs_end_to_end(rows):
    from tests.test_lora import FakeTokenizer, FakeTower

    picked = [r for r in rows if r["split"] == "train"][:16]
    dev = [r for r in rows if r["split"] == "exam1"][:2]
    all_rows = picked + augment.augment(picked, seed=11)
    all_rows += [{**r, "split": f"dev_{r['augment']}"} for r in augment.augment(dev, seed=23, templates=augment.TEST_TEMPLATES)]
    for split in ("exam1", "exam2", "exam3"):
        all_rows += [r for r in rows if r["split"] == split][:1]
    parts = lora_tagger.assign_parts(lora_tagger.encode(all_rows, FakeTokenizer(), "multi"))
    assert parts.get("dev_words") and parts.get("dev_emit")
    tower = FakeTower()
    head, op_head = lora_tagger.Tagger(2, 16), lora_tagger.OpHead(2, 16)
    readers = lora_tagger.make_readers(2, 16)
    result = lora_tagger.train_lora(parts, tower, [tower.adapter], head, [1, 2], "cpu", epochs=1, accumulate=4,
                                    use_amp=False, op_head=op_head, readers=readers, log=lambda *_: None)
    best = result.pop("_best")
    assert len(best["heads"]) == 2 + 4
    assert "dev_words" in result["values"] and "dev_emit" in result["values"]
    assert "/" in result["values"]["dev_emit"]["emit_number_within_1pct"]
    text = lora_tagger.summarize({"model": "fake", "results": {"emit": result}})
    assert "Augmented dev notes" in text


def test_reader_losses_do_not_reach_the_adapters(rows):
    from tests.test_lora import FakeTokenizer, FakeTower

    picked = augment.augment([r for r in rows if r["split"] == "train"][:4], seed=11)
    examples = lora_tagger.encode(picked, FakeTokenizer(), "multi")
    ex = next(e for e in examples if e["op"] == "emit")
    tower = FakeTower()
    readers = lora_tagger.make_readers(2, 16)
    stacked = lora_tagger._stack(tower, ex["ids"], [1, 2])
    doc, question, end = lora_tagger._field_states(stacked, ex)
    loss = lora_tagger._reader_losses(readers, ex, doc[0].detach(), end[0].detach())
    loss.backward()
    assert tower.adapter.grad is None
    assert any(p.grad is not None for p in readers["emit_number"].parameters()) or ex["type"] == "date"
