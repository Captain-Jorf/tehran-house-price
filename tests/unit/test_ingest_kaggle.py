"""Tests for Kaggle data ingestion."""

from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
from tehran_house_price.data import ingest_kaggle


def test_list_csv_files_returns_sorted_csvs(tmp_path: Path) -> None:
    (tmp_path / "b.csv").write_text("x\n", encoding="utf-8")
    (tmp_path / "a.csv").write_text("x\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignore\n", encoding="utf-8")

    files = ingest_kaggle._list_csv_files(tmp_path)

    assert [path.name for path in files] == ["a.csv", "b.csv"]


def test_copy_files_copies_nested_files_to_target_root(tmp_path: Path) -> None:
    src_dir = tmp_path / "src"
    dst_dir = tmp_path / "dst"
    nested_dir = src_dir / "nested"

    nested_dir.mkdir(parents=True)
    dst_dir.mkdir()

    (nested_dir / "housePrice.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (src_dir / "readme.txt").write_text("hello\n", encoding="utf-8")

    copied = ingest_kaggle._copy_files(src_dir, dst_dir)

    assert set(copied) == {"housePrice.csv", "readme.txt"}
    assert (dst_dir / "housePrice.csv").exists()
    assert (dst_dir / "readme.txt").exists()


def test_clean_target_removes_contents_but_keeps_gitkeep(tmp_path: Path) -> None:
    target = tmp_path / "kaggle"
    nested = target / "nested"

    nested.mkdir(parents=True)
    (target / ".gitkeep").write_text("", encoding="utf-8")
    (target / "old.csv").write_text("stale\n", encoding="utf-8")
    (nested / "deep.csv").write_text("old\n", encoding="utf-8")

    ingest_kaggle._clean_target(target)

    remaining = {path.name for path in target.iterdir()}
    assert remaining == {".gitkeep"}


def test_download_dataset_returns_existing_target_without_kaggle_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "kaggle"
    target.mkdir()
    (target / "housePrice.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    called = {"setup": False}

    def fake_target_dir() -> Path:
        return target

    def fake_setup_kaggle_env() -> None:
        called["setup"] = True

    monkeypatch.setattr(ingest_kaggle, "_target_dir", fake_target_dir)
    monkeypatch.setattr(ingest_kaggle, "_setup_kaggle_env", fake_setup_kaggle_env)

    result = ingest_kaggle.download_dataset(force=False)

    assert result == target
    assert called["setup"] is False


def test_download_dataset_raises_runtimeerror_on_403(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    target = tmp_path / "kaggle"
    target.mkdir()

    def fake_target_dir() -> Path:
        return target

    def fake_setup_kaggle_env() -> None:
        return None

    fake_kagglehub = types.ModuleType("kagglehub")

    def raise_forbidden(_dataset: str) -> str:
        raise Exception("403 Forbidden")

    fake_kagglehub.dataset_download = raise_forbidden

    monkeypatch.setattr(ingest_kaggle, "_target_dir", fake_target_dir)
    monkeypatch.setattr(ingest_kaggle, "_setup_kaggle_env", fake_setup_kaggle_env)
    monkeypatch.setitem(sys.modules, "kagglehub", fake_kagglehub)

    with pytest.raises(RuntimeError, match="automatic download failed"):
        ingest_kaggle.download_dataset(force=True)


def test_main_returns_zero_on_success(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    target = Path("data/raw/kaggle")

    def fake_download_dataset(force: bool = False) -> Path:
        assert force is False
        return target

    monkeypatch.setattr(ingest_kaggle, "download_dataset", fake_download_dataset)

    exit_code = ingest_kaggle.main([])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert str(target) in captured.out


def test_main_returns_one_on_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_download_dataset(force: bool = False) -> Path:
        raise RuntimeError("boom")

    monkeypatch.setattr(ingest_kaggle, "download_dataset", fake_download_dataset)

    exit_code = ingest_kaggle.main([])

    assert exit_code == 1
