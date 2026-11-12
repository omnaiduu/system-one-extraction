"""Plain tables from the experiment JSON."""

from __future__ import annotations

BUCKETS = ("exact", "partial", "wrong_place", "missed", "false_alarm")


def _row(name: str, scored: dict) -> str:
    counts = scored.get("buckets", {})
    cells = [f"{counts.get(b, 0)}" for b in BUCKETS]
    return f"| {name} | {scored['exact']}/{scored['fields']} | " + " | ".join(cells) + " |"


def summarize(payload: dict) -> str:
    lines = [
        "# Question order experiment",
        "",
        f"Model `{payload.get('model')}`, hidden states {payload.get('layers', {}).get('layers')} "
        f"of {payload.get('layers', {}).get('depth')}. Run time {payload.get('seconds')} s.",
        "",
        "Evidence = the bracketed words for each field. A field is exact only when the whole set of spans matches.",
        "Cross-validation is 5 folds over the 208 training notes. Dev is exams 1-3, scored once. Exam 4 is not scored.",
        "",
        "| Layout | Fields exact | " + " | ".join(BUCKETS) + " |",
        "| --- | --- | " + " | ".join("---" for _ in BUCKETS) + " |",
    ]
    results = payload["results"]
    for prompt, outcome in results.items():
        lines.append(_row(f"{prompt}, CV (epoch {outcome['cv']['best_epoch']})", outcome["cv"]["at_best"]))
        lines.append(_row(f"{prompt}, dev exams 1-3", outcome["dev"]["all"]))
    lines += ["", "## By operation, dev exams 1-3 (exact / fields)", ""]
    ops = sorted({op for outcome in results.values() for op in outcome["dev"]["all"]["by_op"]})
    lines.append("| Operation | " + " | ".join(results) + " |")
    lines.append("| --- | " + " | ".join("---" for _ in results) + " |")
    for op in ops:
        cells = []
        for outcome in results.values():
            counts = outcome["dev"]["all"]["by_op"].get(op, {})
            cells.append(f"{counts.get('exact', 0)}/{sum(counts.values())}")
        lines.append(f"| {op} | " + " | ".join(cells) + " |")
    lines += ["", "## Check 1: read an amount's digits from one vector (CV over training numbers)", ""]
    lines.append("| Layout | Numbers | Most-common-digit baseline | Best layer, CV exact | Train exact |")
    lines.append("| --- | --- | --- | --- | --- |")
    for prompt, outcome in results.items():
        probe = outcome.get("probe_read_both", {})
        if probe.get("skipped") or not probe.get("layers"):
            continue
        best = max(probe["layers"].items(), key=lambda kv: kv[1]["cv_exact"])
        lines.append(
            f"| {prompt} | {probe['numbers']} | {probe['baseline_exact']} | "
            f"layer {best[0]}: {best[1]['cv_exact']} | {best[1]['train_exact_mean']:.0f} |"
        )
    return "\n".join(lines) + "\n"
