from __future__ import annotations

import json
import pytest

from utils.data_loader import load_interview_data


def test_load_interview_data_reads_dict_payload(tmp_path) -> None:
    payload = {"position": "ADAS Developer", "jd": 1}
    source = tmp_path / "interview_data.json"
    source.write_text(json.dumps(payload), encoding="utf-8")

    assert load_interview_data(source) == payload


def test_load_interview_data_raises_file_not_found(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_interview_data(tmp_path / "missing.json")


def test_load_interview_data_rejects_non_object_root(tmp_path) -> None:
    source = tmp_path / "bad.json"
    source.write_text("[1, 2, 3]", encoding="utf-8")

    with pytest.raises(ValueError, match="Interview data root must be a JSON object"):
        load_interview_data(source)


def test_load_interview_data_raises_json_decode_error(tmp_path) -> None:
    source = tmp_path / "broken.json"
    source.write_text("{not valid JSON}", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        load_interview_data(source)
