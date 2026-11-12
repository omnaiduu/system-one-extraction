from datetime import date
from decimal import Decimal

import pytest
import torch

from experiments import chains, lora_tagger
from experiments.evidence_data import field_evidence, field_operation, load_rows, questions
from experiments.values import ALLOWED, BLOCKS, answer, apply_block, is_rate, matches, run_chain


@pytest.fixture(scope="module")
def rows():
    return [r for r in load_rows() if r["split"] != "exam4"]


def _calc_fields(variants):
    for row in variants:
        for name, field in row["fields"].items():
            if name.startswith("calc_"):
                yield row, name, field


def test_blocks_are_exact():
    assert apply_block("minus_percent", Decimal("2400"), Decimal("15")) == Decimal("2040")
    assert apply_block("minus_percent", Decimal("15"), Decimal("2400"), acc_rate=True) == Decimal("2040")
    assert apply_block("plus_percent", Decimal("2040"), Decimal("18")) == Decimal("2407.2")
    assert apply_block("add_weeks", Decimal("3"), date(2026, 6, 4)) == date(2026, 6, 25)
    assert apply_block("minus_days", date(2026, 6, 25), Decimal("5")) == date(2026, 6, 20)
    assert run_chain(["minus_percent", "plus_percent"], [Decimal("2400"), Decimal("15"), Decimal("18")],
                     [False, True, True]) == Decimal("2407.2")
    assert run_chain(["add"], [Decimal(1)]) is None  # two blocks need three parts
    with pytest.raises(ValueError):
        apply_block("add_days", Decimal("3"), Decimal("4"))
    assert is_rate("charged at 18% on", 13) and is_rate("18 per cent", 2) and not is_rate("18 units", 2)


def test_every_computed_field_reproduces_with_gold_marks_and_gold_blocks(rows):
    """Gold parts folded through the gold blocks give the gold answer, for every template and many seeds."""
    train = [r for r in rows if r["split"] == "train"]
    dev = [r for r in rows if r["split"] in ("exam1", "exam2", "exam3")]
    variants = chains.training_variants(train, seed=11) + chains.test_variants(dev, seed=23)
    variants += chains.training_variants(train[:40], seed=99, copies=3)
    checked, kinds = 0, set()
    for row, name, field in _calc_fields(variants):
        spans = field_evidence(field)
        parts = [row["text"][a:b] for a, b in spans]
        assert [e["text"] for e in field["evidence"]] == parts  # the spans sit on the inserted words
        op = field_operation(field)
        assert op in ALLOWED[field["type"]]
        rates = [is_rate(row["text"], b, p) for (a, b), p in zip(spans, parts)]
        predicted = answer(op, parts, field, blocks=field.get("blocks"), rates=rates)
        assert matches(field, predicted), (row["id"], field["chain_kind"], parts, field["value"], predicted)
        kinds.add(field["chain_kind"])
        checked += 1
    assert checked > 1000
    all_kinds = {t[0] for templates in chains.TEMPLATE_SETS.values() for t in templates}
    assert kinds == all_kinds


def test_answers_are_never_printed_and_names_do_not_leak(rows):
    from extractor.audit import number_present

    variants = chains.test_variants([r for r in rows if r["split"] == "exam1"], seed=23)
    for row, name, field in _calc_fields(variants):
        assert name in ("calc_0", "calc_1")
        if field["type"] == "number":
            assert not number_present(row["text"], Decimal(str(field["value"])), 2)
        else:
            assert field["value"] not in row["text"]
        # Every original field is untouched apart from its offsets.
        source = next(r for r in rows if r["id"] == row["id"].split("~")[0])
        for other in source["fields"]:
            assert row["fields"][other]["type"] == source["fields"][other]["type"]
            assert row["fields"][other].get("value") == source["fields"][other].get("value")


def test_unseen_combinations_are_unseen_and_their_blocks_are_seen():
    train_combos = {tuple(t[1]) for t in chains.TRAIN_COMBO}
    unseen = {tuple(t[1]) for t in chains.TEST_COMBO_UNSEEN}
    seen = {tuple(t[1]) for t in chains.TEST_COMBO_SEEN}
    assert not unseen & train_combos
    assert seen <= train_combos
    trained_blocks = {b for t in chains.TRAIN_SINGLE + chains.TRAIN_COMBO if t[1] != "average" for b in t[1]}
    assert {b for combo in unseen for b in combo} <= trained_blocks
    train_text = {t[2] for t in chains.TRAIN_SINGLE + chains.TRAIN_COMBO}
    train_instr = {t[3] for t in chains.TRAIN_SINGLE + chains.TRAIN_COMBO}
    for t in chains.TEST_SINGLE + chains.TEST_COMBO_SEEN + chains.TEST_COMBO_UNSEEN:
        assert t[2] not in train_text and t[3] not in train_instr


def test_step_head_learns_a_planted_signal():
    """A block that is written into the state after each operand is learnable."""
    torch.manual_seed(0)
    head = lora_tagger.StepHead(1, 8, inner=32, window=1)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    picks = [BLOCKS.index("minus_percent"), BLOCKS.index("plus_percent")]

    def sample(seed):
        g = torch.Generator().manual_seed(seed)
        doc = torch.randn(1, 12, 8, generator=g) * 0.1
        labels = [picks[int(torch.randint(0, 2, (1,), generator=g))] for _ in range(2)]
        for k, label in enumerate(labels):
            doc[0, 4 * (k + 1) + 1, picks.index(label)] += 3.0  # the word after operand k+1
        return doc, labels

    for step in range(300):
        doc, labels = sample(step)
        loss = torch.nn.functional.cross_entropy(head(doc, doc[:, -1], [(0, 0), (4, 4), (8, 8)]), torch.tensor(labels))
        opt.zero_grad()
        loss.backward()
        opt.step()
    head.eval()
    right = 0
    for seed in range(1000, 1050):
        doc, labels = sample(seed)
        right += head(doc, doc[:, -1], [(0, 0), (4, 4), (8, 8)]).argmax(-1).tolist() == labels
    assert right >= 45
    assert head(torch.zeros(1, 5, 8), torch.zeros(1, 8), [(0, 0)]) is None


def test_chain_training_runs_end_to_end(rows):
    from tests.test_lora import FakeTokenizer, FakeTower

    picked = [r for r in rows if r["split"] == "train"][:16]
    dev = [r for r in rows if r["split"] == "exam1"][:2]
    all_rows = picked + chains.training_variants(picked, seed=11, copies=1)
    all_rows += [{**r, "split": f"dev_{r['chain_test']}"} for r in chains.test_variants(dev, seed=23)]
    for split in ("exam1", "exam2", "exam3"):
        all_rows += [r for r in rows if r["split"] == split][:1]
    examples = lora_tagger.encode(all_rows, FakeTokenizer(), "multi")
    chain_ex = [e for e in examples if e["op"] == "chain"]
    assert chain_ex and all(len(e["blocks"]) == len(e["gold"]) - 1 for e in chain_ex)
    parts = lora_tagger.assign_parts(examples)
    assert parts.get("dev_single") and parts.get("dev_combo_seen") and parts.get("dev_combo_unseen")
    tower = FakeTower()
    head, op_head, step_head = lora_tagger.Tagger(2, 16), lora_tagger.OpHead(2, 16), lora_tagger.StepHead(2, 16)
    result = lora_tagger.train_lora(parts, tower, [tower.adapter], head, [1, 2], "cpu", epochs=1, accumulate=4,
                                    use_amp=False, op_head=op_head, step_head=step_head, log=lambda *_: None)
    best = result.pop("_best")
    assert len(best["heads"]) == 3
    unseen = result["values"]["dev_combo_unseen"]
    assert unseen["by_steps"] and "/" in unseen["blocks_given_right_marks"]
    text = lora_tagger.summarize({"model": "fake", "results": {"chains": result}})
    assert "Computed fields" in text and "dev_combo_unseen" in text


def test_step_head_gradient_reaches_the_adapters(rows):
    from tests.test_lora import FakeTokenizer, FakeTower

    picked = chains.training_variants([r for r in rows if r["split"] == "train"][:4], seed=11, copies=1)
    ex = next(e for e in lora_tagger.encode(picked, FakeTokenizer(), "multi") if e["op"] == "chain")
    tower = FakeTower()
    step_head = lora_tagger.StepHead(2, 16)
    stacked = lora_tagger._stack(tower, ex["ids"], [1, 2])
    doc, question, end = lora_tagger._field_states(stacked, ex)
    logits = step_head(doc[0], end[0], ex["gold"])
    torch.nn.functional.cross_entropy(logits, torch.tensor(ex["blocks"])).backward()
    assert tower.adapter.grad is not None


def test_old_checkpoints_keep_their_operation_indexes():
    from experiments.values import OPS

    assert OPS[:13] == ("copy", "list", "read_number", "read_date", "read_time", "read_datetime",
                        "sum", "subtract", "product", "add_days", "days_between", "median", "emit")
    old = lora_tagger.OpHead(2, 16, outputs=13)
    ex = {"type": "number"}
    assert lora_tagger.choose_op(ex, torch.zeros(13)) in ALLOWED["number"][:5]


def test_final_grade_summary_reports_computed_fields():
    from experiments.kaggle_final_job import summarize

    sel = {"auroc": 0.5, "thresholds": {"0.9": {"kept": 1, "right": 1}}, "fields": 2}
    report = {"exact": 3, "fields": 5, "by_op": {}, "op_choice": "1/1", "confidence": sel,
              "blocks_given_right_marks": "2/3", "by_kind": {"split": "1/2"}, "by_steps": {"1": "1/2"}}
    payload = {"source": "x", "layout": "multi", "bias": 0.0,
               "exam4": {"values": report, "evidence": {"exact": 1, "fields": 2}},
               "exam4_chains": {"combo_unseen": {"computed": report, "all_fields": report}},
               "sroie": {"receipts": 1, "skipped": {}, "score": {
                   "overall": {"fields": 1, "found": 1, "value": 1, "value_wide_dates": 1},
                   "by_field": {}, "confidence": sel}}}
    text = summarize(payload)
    assert "combo_unseen | 3/5 | 2/3" in text


def test_old_op_head_without_new_operations():
    """An operation head trained before "average" and "chain" existed has 12 outputs; nothing may index past them."""
    import torch

    from experiments.lora_tagger import choose_op, field_confidence, legal_ops
    from experiments.values import OPS

    ex = {"type": "number"}
    op_logits = torch.zeros(OPS.index("average"))
    assert "average" not in legal_ops(ex, op_logits) and "chain" not in legal_ops(ex, op_logits)
    op = choose_op(ex, op_logits)
    tags = torch.zeros(6, 3)
    tags[2, 1] = 5.0
    conf = field_confidence(ex, tags, op_logits, 0.0, [(2, 3)], op)
    assert 0.0 < conf["confidence"] <= 1.0
