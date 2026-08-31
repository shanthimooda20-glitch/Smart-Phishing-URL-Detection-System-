"""Tests for the dataset-loading and preparation stages of the pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.ml.train import (
    build_feature_matrix,
    drop_constant_features,
    load_dataset,
)


def _write_csv(path: Path, rows: str) -> Path:
    path.write_text(rows, encoding="utf-8")
    return path


def test_load_dataset_accepts_alternative_column_names(tmp_path: Path):
    """The loader detects url/label columns case-insensitively."""
    csv = _write_csv(
        tmp_path / "data.csv",
        "URL,isMalicious\nexample.com/a,0\nevil.tk/login,1\n",
    )
    frame = load_dataset(csv)
    assert list(frame.columns) == ["url", "label"]
    assert set(frame["label"]) == {0, 1}


def test_load_dataset_maps_textual_labels(tmp_path: Path):
    """Textual labels such as phishing/benign are mapped onto 1/0."""
    csv = _write_csv(
        tmp_path / "data.csv",
        "url,type\nexample.com/a,benign\nevil.tk/login,phishing\nx.com,legitimate\n",
    )
    frame = load_dataset(csv)
    assert frame["label"].tolist() == [0, 1, 0]


def test_load_dataset_cleans_dirty_rows(tmp_path: Path):
    """Blank URLs, unusable labels and duplicates are removed."""
    csv = _write_csv(
        tmp_path / "data.csv",
        "url,label\n"
        "example.com/a,0\n"
        "example.com/a,0\n"      # duplicate
        ",1\n"                    # empty URL
        "short,1\n"               # kept: >= 4 characters
        "weird.com,maybe\n"       # unusable label
        "evil.tk/login,1\n",
    )
    frame = load_dataset(csv)
    assert len(frame) == 3
    assert frame["url"].is_unique


def test_load_dataset_reports_a_missing_file(tmp_path: Path):
    """A missing dataset produces an actionable error."""
    with pytest.raises(FileNotFoundError) as error:
        load_dataset(tmp_path / "absent.csv")
    assert "data/README.md" in str(error.value)


def test_load_dataset_rejects_a_single_class_corpus(tmp_path: Path):
    """Training on one class is refused rather than silently useless."""
    csv = _write_csv(tmp_path / "data.csv", "url,label\na.com/x,0\nb.com/y,0\n")
    with pytest.raises(ValueError):
        load_dataset(csv)


def test_load_dataset_rejects_missing_columns(tmp_path: Path):
    """A CSV without the required columns fails fast with an explanation."""
    csv = _write_csv(tmp_path / "data.csv", "foo,bar\n1,2\n")
    with pytest.raises(ValueError) as error:
        load_dataset(csv)
    assert "must contain a URL column" in str(error.value)


def test_feature_matrix_matches_the_declared_schema():
    """The training matrix is finite and uses the canonical column order."""
    matrix = build_feature_matrix(["example.com/a", "evil.tk/login?x=1", ""])
    assert len(matrix) == 3
    assert matrix.notna().all().all()
    assert matrix.select_dtypes("number").shape[1] == matrix.shape[1]


def test_constant_columns_are_dropped():
    """Zero-variance features carry no signal and must be removed."""
    frame = pd.DataFrame({"varies": [1, 2, 3], "constant": [7, 7, 7]})
    reduced, dropped = drop_constant_features(frame)
    assert dropped == ["constant"]
    assert list(reduced.columns) == ["varies"]
