"""Harness regression checks; not part of the student's scored criteria."""

import json
from unittest.mock import patch

import pytest

from grade import RUBRIC
from grading.performance import write_result


def test_paired_median_resists_one_timing_outlier(tmp_path):
    output = tmp_path / "result.json"
    with patch("torch.cuda.get_device_name", return_value="Tesla V100"):
        write_result("dp", [2, 2, 100, 2, 2], [1, 1, 100, 1, 1], output)
    report = json.loads(output.read_text())
    assert report["median_speedup"] == 2
    assert report["paired_speedups"] == [2, 2, 1, 2, 2]
    assert report["passed"]


def test_slow_implementation_fails_but_preserves_measurements(tmp_path):
    output = tmp_path / "result.json"
    with (
        patch("torch.cuda.get_device_name", return_value="Tesla V100"),
        pytest.raises(AssertionError, match="median speedup"),
    ):
        write_result("pipeline", [1] * 5, [2] * 5, output)
    report = json.loads(output.read_text())
    assert report["median_speedup"] == 0.5
    assert not report["passed"]
    assert report["parallel_seconds_per_step"] == [2] * 5


def test_rubric_keeps_cd_correctness_and_total():
    assert sum(row[2] for row in RUBRIC) == 100
    performance = [row for row in RUBRIC if row[3] == "performance"]
    assert [(row[1], row[2]) for row in performance] == [("dp", 10), ("pipeline", 10)]
    assert sum(row[2] for row in RUBRIC if row[1] == "finetune") == 25
    assert sum(row[2] for row in RUBRIC if row[1] == "inference") == 20
