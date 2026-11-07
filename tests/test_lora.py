import re

import torch
from torch import nn

from experiments import lora_tagger
from experiments.evidence_data import load_rows
from experiments.evidence_heads import Tagger


class FakeTokenizer:
    def __call__(self, text, return_offsets_mapping=True, add_special_tokens=True):
        offsets = [(m.start(), m.end()) for m in re.finditer(r"\s*\S+", text)]
        ids = [sum(map(ord, text[a:b].strip())) % 500 + 3 for a, b in offsets]
        if add_special_tokens:
            ids, offsets = [2] + ids, [(0, 0)] + offsets
        return {"input_ids": ids, "offset_mapping": offsets}


class FakeTower(nn.Module):
    """Embedding plus two layers; returns a hidden-state tuple like HF models."""

    def __init__(self, width=16):
        super().__init__()
        self.embed = nn.Embedding(600, width)
        self.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(2)])
        self.requires_grad_(False)
        self.adapter = nn.Parameter(torch.zeros(width))

    def forward(self, ids):
        h = self.embed(torch.tensor([ids])) + self.adapter
        out = [h]
        for layer in self.layers:
            h = torch.tanh(layer(h)) + h
            out.append(h)
        return tuple(out)


def _parts():
    rows = load_rows()
    picked = [r for r in rows if r["split"] == "train"][:20]
    for split in ("exam1", "exam2", "exam3"):
        picked += [r for r in rows if r["split"] == split][:1]
    return lora_tagger.assign_parts(lora_tagger.encode(picked, FakeTokenizer(), "first"))


def test_encode_and_split():
    parts = _parts()
    assert parts["fit"] and parts["held"] and all(parts[d] for d in lora_tagger.DEV)
    fit_docs = {ex["doc"] for ex in parts["fit"]}
    assert not fit_docs & {ex["doc"] for ex in parts["held"]}
    for ex in parts["fit"]:
        assert len(ex["tags"]) == len(ex["doc_index"])
        assert ex["q_index"] < ex["doc_index"][0]  # question before the note


def test_bias_moves_the_cut_off():
    logits = torch.tensor([[2.0, 1.5, 0.0], [2.0, 0.0, 1.5], [0.0, -1.0, -1.0]])
    assert lora_tagger.decode(logits, 0.0) == []
    assert lora_tagger.decode(logits, 1.0) == [(0, 1)]


def test_frozen_and_lora_runs_end_to_end():
    parts = _parts()
    tower = FakeTower()
    layers = [1, 2]
    frozen = lora_tagger.run_frozen(parts, tower, layers, "cpu", epochs=2, every=1, log=lambda *_: None)
    assert frozen["epoch"] in (1, 2)
    head = Tagger(len(layers), 16)
    before = tower.adapter.detach().clone()
    result = lora_tagger.train_lora(parts, tower, [tower.adapter], head, layers, "cpu", epochs=2,
                                    accumulate=2, use_amp=False, log=lambda *_: None)
    assert not torch.equal(before, result["_best"]["adapter"][0]) or result["epoch"] == 1
    result.pop("_best")
    text = lora_tagger.summarize({"model": "fake", "results": {"frozen first": frozen, "lora first": result}})
    assert "| lora first |" in text


def test_peft_injects_only_text_attention():
    from peft import LoraConfig, inject_adapter_in_model

    from experiments.kaggle_lora_job import attention_targets

    class Attn(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj, self.k_proj, self.v_proj, self.o_proj = (nn.Linear(8, 8) for _ in range(4))

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attn()
            self.mlp = nn.Linear(8, 8)

    class Lang(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer(), Layer()])

    class Vision(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer()])

    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.language_model = Lang()
            self.vision_tower = Vision()

    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

    model = Outer()
    targets = attention_targets(model)
    assert len(targets) == 8 and all("language_model" in t for t in targets)
    model = inject_adapter_in_model(LoraConfig(r=4, lora_alpha=8, target_modules=targets), model)
    lora_names = [n for n, _ in model.named_parameters() if "lora_" in n]
    assert lora_names and all("language_model" in n for n in lora_names)
