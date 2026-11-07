import random
from datetime import date
from decimal import Decimal

import pytest
import torch

from experiments import augment, chains, lora_tagger, rewrite
from experiments.evidence_data import field_evidence, field_operation, load_rows
from experiments.values import ALLOWED, answer, is_rate, matches, snap_char_spans


@pytest.fixture(scope="module")
def rows():
    return [r for r in load_rows() if r["split"] != "exam4"]


def test_snapping_takes_whole_numbers_and_dates():
    text = "The 3,720.78 fee rises by 12% next year. Delivery is 4 weeks after 25 October 2027. Ref HS-26/4418."
    at = text.index("3,720")
    assert [text[a:b] for a, b in snap_char_spans(text, [(at, at + 1)], "number")] == ["3,720.78"]
    oc = text.index("October")
    assert [text[a:b] for a, b in snap_char_spans(text, [(oc, oc + 4)], "date")] == ["25 October 2027"]
    # Two pieces of one number become one mark; names without digits are untouched.
    assert snap_char_spans(text, [(at, at + 2), (at + 3, at + 6)], "number") == [(at, at + 8)]
    assert snap_char_spans("Adriatic Pipeworks", [(0, 8)], "number") == [(0, 8)]
    # A number field does not swallow a date around a marked day count.
    four = text.index("4 weeks")
    assert [text[a:b] for a, b in snap_char_spans(text, [(four, four + 1)], "number")] == ["4"]


def test_backwards_check_accepts_the_right_value_and_only_it():
    rng = random.Random(0)
    for _ in range(1500):
        value = Decimal(rng.randint(0, 9_999_999)) / 100
        said = augment.number_words(value, 2, rng)
        assert rewrite.agrees(said, value, "number"), said
        assert not rewrite.agrees(said, value + Decimal("0.01"), "number")
        assert not rewrite.agrees(said, value + 1, "number")
    for _ in range(800):
        d = date(2024, 1, 1) + __import__("datetime").timedelta(days=rng.randint(0, 2000))
        said = augment.date_words(d, rng)
        assert rewrite.agrees(said, d, "date"), said
        assert not rewrite.agrees(said, d + __import__("datetime").timedelta(days=1), "date")
    assert rewrite.agrees("Rupees One Lakh Twenty Thousand Only", Decimal(120000), "number")
    assert rewrite.agrees("4th June 2026", date(2026, 6, 4), "date")
    assert not rewrite.agrees("1,486.75", Decimal("1486.75"), "number")  # digits: nothing to check against


def test_constrained_decoding_only_emits_allowed_tokens():
    vocab = ["<eos>", "a", "1", "2", "0", ".", "-", "hello"]
    allowed = [2, 3, 4, 5, 6]
    script = [7, 2, 4, 3, 1, 0]  # the fake model's favourite token at each step

    def next_logits(ids):
        step = len(ids) - 3
        logits = torch.zeros(len(vocab))
        logits[script[step]] = 10.0
        logits[7] = 20.0  # "hello" is best every time but not allowed
        return logits

    out = rewrite.greedy_constrained(next_logits, [1, 1, 1], allowed, {0})
    assert all(t in allowed for t in out)
    assert rewrite.parse_output("120000", "number") == Decimal(120000)
    assert rewrite.parse_output("2026-04-02", "date") == date(2026, 4, 2)
    assert rewrite.parse_output("2026-13-02", "date") is None and rewrite.parse_output("1.5", "count") is None


def test_chain_reads_words_through_the_reader():
    field = {"type": "number", "decimals": 2}
    got = answer("chain", ["one thousand two hundred", "15"], field, blocks=["minus_percent"], rates=[False, True],
                 read=lambda i, kind: Decimal(1200) if i == 0 else None)
    assert got["value"] == 1020.0
    assert answer("chain", ["one thousand", "15"], field, blocks=["minus_percent"], rates=[False, True])["value"] is None


def test_every_grammar_field_reproduces(rows):
    train = [r for r in rows if r["split"] == "train"]
    dev = [r for r in rows if r["split"] in ("exam1", "exam2", "exam3")]
    variants = chains.grammar_training_variants(train, seed=11) + chains.grammar_test_variants(dev, seed=23)
    variants += chains.grammar_training_variants(train[:60], seed=77, copies=4)
    checked, train_pairs, unseen = 0, set(), set()
    for row in variants:
        for name, field in row["fields"].items():
            if not name.startswith("calc_"):
                continue
            spans = field_evidence(field)
            parts = [row["text"][a:b] for a, b in spans]
            assert [e["text"] for e in field["evidence"]] == parts
            op = field_operation(field)
            assert op in ALLOWED[field["type"]]
            rates = [is_rate(row["text"], b, p) for (a, b), p in zip(spans, parts)]
            assert matches(field, answer(op, parts, field, blocks=field.get("blocks"), rates=rates)), (field, parts)
            if field["chain_set"] == "train_combo":
                train_pairs.add(tuple(field["blocks"]))
            if field["chain_set"].startswith("combo_unseen"):
                unseen.add(tuple(field["blocks"]))
            checked += 1
    assert checked > 1500
    assert not train_pairs & chains.HELD_OUT_PAIRS and unseen <= chains.HELD_OUT_PAIRS
    # Every block appears first and second in training pairs.
    for block in chains.NUMBER_STEP_BLOCKS + chains.DATE_STEP_BLOCKS:
        assert any(p[1] == block for p in train_pairs)
        assert block in chains.LAST_ONLY or any(p[0] == block for p in train_pairs)


def test_grammar_wordings_are_split():
    for table in (chains.G_OPEN_NUMBER, chains.G_OPEN_DATE, chains.G_JOIN, chains.G_ASK_NUMBER, chains.G_ASK_DATE, chains.G_AVERAGE):
        assert not set(map(str, table["train"])) & set(map(str, table["test"]))
    for block, wordings in chains.G_STEPS.items():
        assert wordings["test"] and not set(wordings["train"]) & set(wordings["test"]), block


def test_one_operand_step_head_and_snapped_training_run(rows):
    from tests.test_lora import FakeTokenizer, FakeTower

    head = lora_tagger.StepHead(2, 16, pair_context=False, step_embedding=False)
    assert head.step is None and head(torch.randn(2, 9, 16), torch.randn(2, 16), [(0, 0), (3, 4), (7, 8)]).shape == (2, 11)

    picked = [r for r in rows if r["split"] == "train"][:16]
    dev = [r for r in rows if r["split"] == "exam1"][:2]
    all_rows = picked + chains.grammar_training_variants(picked, seed=11, copies=1)
    all_rows += augment.augment(picked[:4], seed=13, words=1, emits=0)
    all_rows += [{**r, "split": f"dev_{r['chain_test']}"} for r in chains.grammar_test_variants(dev, seed=23)]
    all_rows += [{**r, "split": "dev_words"} for r in augment.augment(dev, seed=29, words=1, emits=0, templates=augment.TEST_TEMPLATES)]
    for split in ("exam1", "exam2", "exam3"):
        all_rows += [r for r in rows if r["split"] == split][:1]
    parts = lora_tagger.assign_parts(lora_tagger.encode(all_rows, FakeTokenizer(), "multi"))
    assert parts.get("dev_combo_unseen_familiar") and parts.get("dev_words")
    asked = []

    def fake_rewriter(text, kind):
        asked.append((text, kind))
        return None

    tower = FakeTower()
    result = lora_tagger.train_lora(parts, tower, [tower.adapter], lora_tagger.Tagger(2, 16), [1, 2], "cpu",
                                    epochs=1, accumulate=4, use_amp=False, op_head=lora_tagger.OpHead(2, 16),
                                    step_head=head, snap=True, rewriter=fake_rewriter, log=lambda *_: None)
    result.pop("_best")
    assert set(result["values_plain"]) == set(result["values"]) == set(result["values_no_rewrite"])
    text = lora_tagger.summarize({"model": "fake", "results": {"grammar": result}, "rewrite_stats": {"asked": len(asked)}})
    assert "Effect of snapping" in text and "dev_combo_unseen_familiar" in text
