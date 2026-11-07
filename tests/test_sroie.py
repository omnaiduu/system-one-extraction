import json

import pytest

from experiments import sroie
from extractor.audit import Drop

BOX = """\
98,26,321,26,321,66,98,66,TAN CHAY YEE
80,119,329,119,329,140,80,140,OJC MARKETING SDN BHD
104,163,306,163,306,182,104,182,NO 2 & 4, JALAN BAYU 4,
123,185,286,185,286,205,123,205,BANDAR SERI ALAM,
116,205,292,205,292,223,116,223,81750 MASAI, JOHOR
27,326,109,326,109,341,27,341,DATE : 15/01/2019 11:05:16
27,500,109,500,109,520,27,520,SUB TOTAL : 193.00
27,540,109,540,109,560,27,560,TOTAL: 193.00
27,580,109,580,109,600,27,600,CASH 200.00, CHANGE 7.00
"""


def test_read_box_keeps_commas_inside_the_text():
    lines = sroie.read_box(BOX)
    assert lines[2] == "NO 2 & 4, JALAN BAYU 4,"
    assert len(lines) == 9


def test_build_row_locates_labels_and_skips_typos():
    entities = {"company": "OJC MARKETING SDN BHD", "date": "15/01/2019",
                "address": "NO 2 & 4, JALAN BAYU 4, BANDAR SERI ALAM, 81750 MASAI, JOHOR", "total": "RM 193.00"}
    row, skipped = sroie.build_row("X1", BOX, entities)
    text = row["text"]
    f = row["fields"]
    assert text[f["company"]["start"]:f["company"]["end"]] == "OJC MARKETING SDN BHD"
    assert "BANDAR SERI ALAM,\n81750 MASAI, JOHOR" in text[f["address"]["start"]:f["address"]["end"]]
    assert f["date"]["value"] == "2019-01-15"
    assert f["total"]["value"] == 193.0
    # The TOTAL line wins over SUB TOTAL.
    assert "TOTAL: 193.00" in text[text.rfind("\n", 0, f["total"]["start"]) + 1 : f["total"]["end"]]
    assert "SUB" not in text[text.rfind("\n", 0, f["total"]["start"]) + 1 : f["total"]["end"]]
    assert skipped == []

    typo = dict(entities, address="NO 2 & 4, JALAN BAYU 4, BANDAR SERI ALAM, B1750 MASAI, JOHOR")
    row, skipped = sroie.build_row("X1", BOX, typo)
    assert "address" not in row["fields"] and skipped == ["address:not_in_text"]


def test_punctuation_and_spacing_differences_still_match():
    text = "NO. 3, JALAN SS21/39, 47400 PETALING JAYA\nKLANG SELANGOR D. E"
    assert sroie.locate_text(text, "NO.3, JALAN SS21/39,47400 PETALING JAYA") == (0, text.index("JAYA") + 4)
    assert sroie.locate_text(text, "SELANGOR D.E") is not None


@pytest.mark.parametrize("raw,iso", [
    ("15/01/2019", "2019-01-15"), ("21-03-18", "2018-03-21"), ("01/01/18", "2018-01-01"),
    ("09-JAN-2017", "2017-01-09"), ("03 MAR 19", "2019-03-03"), ("06 APR 2018", "2018-04-06"),
    ("2018-12-25", "2018-12-25"), ("10.03.2018", "2018-03-10"),
])
def test_wide_date_reader(raw, iso):
    assert sroie.read_date_wide(raw).isoformat() == iso


def test_wide_date_reader_refuses_nonsense():
    with pytest.raises(Drop):
        sroie.read_date_wide("31/02/2018")
    with pytest.raises(Drop):
        sroie.read_date_wide("TOTAL")


def test_load_split_counts_missing_files(tmp_path):
    (tmp_path / "box").mkdir()
    (tmp_path / "entities").mkdir()
    (tmp_path / "box" / "A.txt").write_text(BOX)
    (tmp_path / "entities" / "A.txt").write_text(json.dumps({"company": "OJC MARKETING SDN BHD", "total": "193.00"}))
    (tmp_path / "box" / "B.txt").write_text(BOX)
    rows, skipped = sroie.load_split(tmp_path)
    assert len(rows) == 1 and skipped["no_entities_file"] == 1
    assert skipped["date:no_label"] == 1 and skipped["address:no_label"] == 1


def test_only_the_final_kernel_bundles_exam4(tmp_path):
    import base64, io, tarfile
    from experiments.make_kaggle_kernel import write_kernel

    def names(script):
        ns = {}
        exec(script.read_text().split("\n", 1)[0], ns)
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(ns["BUNDLE"])), mode="r:gz") as tar:
            return tar.getnames()

    for job in ("question_order", "lora", "lora_onepass"):
        assert not any(n.startswith("data/exam4") for n in names(write_kernel(tmp_path / job, job)))
    with pytest.raises(ValueError):
        write_kernel(tmp_path / "f0", "final")
    final = write_kernel(tmp_path / "f", "final", "ommnaidu/gemma-e2b-lora-onepass")
    assert any(n.startswith("data/exam4/raw/") for n in names(final))
    meta = json.loads((tmp_path / "f" / "kernel-metadata.json").read_text())
    assert meta["kernel_sources"] == ["ommnaidu/gemma-e2b-lora-onepass"]
    assert meta["dataset_sources"] == ["urbikn/sroie-datasetv2"]
    assert "kaggle_final_job import main as main" in final.read_text()
