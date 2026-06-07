"""Tests for dataset utilities."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from structured_ocr.training import DatasetUtils, PreparedSample
from structured_ocr.training.dataset_utils import INSTRUCTION_PROMPT


def _write_jsonl(path: Path, records: list[dict]) -> None:
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def test_load_jsonl(tmp_path: Path):
    path = tmp_path / "train.jsonl"
    _write_jsonl(
        path,
        [
            {"reference": "\\section{Intro}", "image": "img1.png"},
            {"reference": "\\section{Methods}", "image": "img2.png"},
        ],
    )
    records = DatasetUtils().load(path)
    assert len(records) == 2
    assert records[0]["reference"].startswith("\\section")


def test_load_json_envelope(tmp_path: Path):
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"samples": [{"reference": "x"}]}))
    records = DatasetUtils().load(path)
    assert records == [{"reference": "x"}]


def test_load_csv(tmp_path: Path):
    path = tmp_path / "data.csv"
    path.write_text("reference,image\n\\section{A},a.png\n")
    records = DatasetUtils().load(path)
    assert records[0]["reference"] == "\\section{A}"


def test_load_unsupported_format(tmp_path: Path):
    path = tmp_path / "data.txt"
    path.write_text("nope")
    with pytest.raises(ValueError, match="Unsupported dataset format"):
        DatasetUtils().load(path)


def test_prepare_from_records():
    records = [
        {"reference": "x = 1", "image": "img1.png", "template": "equation"},
        {"latex": "y = 2", "image": "img2.png"},
        {"output": "z = 3"},
    ]
    samples = DatasetUtils().prepare(records)
    assert len(samples) == 3
    assert all(isinstance(s, PreparedSample) for s in samples)
    assert samples[0].reference == "x = 1"
    assert samples[0].image == "img1.png"
    assert "Document type: equation" in samples[0].prompt
    assert "[image: img1.png]" in samples[0].prompt


def test_prepare_skips_empty_references():
    records = [{"reference": ""}, {"output": "x"}]
    samples = DatasetUtils().prepare(records)
    assert len(samples) == 1
    assert samples[0].reference == "x"


def test_split_returns_train_and_eval():
    utils = DatasetUtils(seed=123)
    samples = [
        PreparedSample(prompt=str(i), reference=str(i), image=None, metadata={})
        for i in range(20)
    ]
    train, evals = utils.split(samples, eval_ratio=0.1)
    assert len(train) == 18
    assert len(evals) == 2


def test_split_handles_empty():
    train, evals = DatasetUtils().split([])
    assert train == [] and evals == []


def test_write_jsonl(tmp_path: Path):
    utils = DatasetUtils()
    samples = [PreparedSample(prompt="p", reference="r", image="i.png", metadata={"k": "v"})]
    out = utils.write_jsonl(samples, tmp_path / "out.jsonl")
    assert out.exists()
    with open(out) as f:
        record = json.loads(f.readline())
    assert record["prompt"] == "p"
    assert record["reference"] == "r"
    assert record["image"] == "i.png"
    assert record["metadata"] == {"k": "v"}


def test_write_hf_dataset(tmp_path: Path):
    pytest.importorskip("datasets")
    utils = DatasetUtils()
    samples = [PreparedSample(prompt="p", reference="r", image=None, metadata={})]
    out = utils.write_hf_dataset(samples, tmp_path / "hf")
    assert out.exists()


def test_statistics_basic():
    samples = [
        PreparedSample(prompt="abcdef", reference="x" * 10, image="a.png", metadata={}),
        PreparedSample(prompt="ghi", reference="y" * 4, image=None, metadata={}),
    ]
    stats = DatasetUtils().statistics(samples)
    assert stats["count"] == 2
    assert stats["with_image"] == 1
    assert stats["min_prompt_len"] == 3
    assert stats["max_prompt_len"] == 6


def test_instruction_prompt_default():
    assert "LaTeX" in INSTRUCTION_PROMPT
