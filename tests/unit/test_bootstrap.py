"""Unit tests for the startup artifact bootstrap module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from tehran_house_price.api import bootstrap
from tehran_house_price.api.bootstrap import (
    ArtifactDownloadError,
    ensure_model_artifacts,
)
from tehran_house_price.api.model_loader import (
    DEFAULT_METADATA_FILENAME,
    DEFAULT_MODEL_FILENAME,
)


class _FakeSettings:
    def __init__(
        self,
        artifact_url: str | None = None,
        metadata_url: str | None = None,
    ) -> None:
        self.artifact_download_url = artifact_url
        self.artifact_metadata_download_url = metadata_url


def _patch_settings(monkeypatch, artifact_url=None, metadata_url=None):
    fake = _FakeSettings(artifact_url=artifact_url, metadata_url=metadata_url)
    monkeypatch.setattr(bootstrap, "get_settings", lambda: fake)


def test_noop_when_all_files_present(tmp_path: Path, monkeypatch) -> None:
    """If both artifact files exist on disk, no download is attempted."""
    (tmp_path / DEFAULT_MODEL_FILENAME).write_bytes(b"fake-model")
    (tmp_path / DEFAULT_METADATA_FILENAME).write_text("{}")

    _patch_settings(
        monkeypatch,
        artifact_url="http://example.com/a",
        metadata_url="http://example.com/b",
    )

    with patch.object(bootstrap, "_download_file") as mock_dl:
        ensure_model_artifacts(models_dir=tmp_path)

    mock_dl.assert_not_called()


def test_noop_when_files_missing_and_urls_unset(tmp_path: Path, monkeypatch) -> None:
    """If files are missing and no URLs are configured, log-and-continue."""
    _patch_settings(monkeypatch, artifact_url=None, metadata_url=None)

    with patch.object(bootstrap, "_download_file") as mock_dl:
        ensure_model_artifacts(models_dir=tmp_path)

    mock_dl.assert_not_called()


def test_downloads_missing_files(tmp_path: Path, monkeypatch) -> None:
    """When files are missing and URLs are set, download is invoked."""
    _patch_settings(
        monkeypatch,
        artifact_url="http://example.com/model.joblib",
        metadata_url="http://example.com/meta.json",
    )

    def fake_download(url: str, destination: Path) -> None:
        destination.write_bytes(b"downloaded")

    with patch.object(bootstrap, "_download_file", side_effect=fake_download) as mock_dl:
        ensure_model_artifacts(models_dir=tmp_path)

    assert mock_dl.call_count == 2
    assert (tmp_path / DEFAULT_MODEL_FILENAME).exists()
    assert (tmp_path / DEFAULT_METADATA_FILENAME).exists()


def test_downloads_only_missing_file(tmp_path: Path, monkeypatch) -> None:
    """Only the missing file should be downloaded, not the existing one."""
    (tmp_path / DEFAULT_MODEL_FILENAME).write_bytes(b"already-here")

    _patch_settings(
        monkeypatch,
        artifact_url="http://example.com/model.joblib",
        metadata_url="http://example.com/meta.json",
    )

    def fake_download(url: str, destination: Path) -> None:
        destination.write_bytes(b"downloaded")

    with patch.object(bootstrap, "_download_file", side_effect=fake_download) as mock_dl:
        ensure_model_artifacts(models_dir=tmp_path)

    assert mock_dl.call_count == 1
    called_dest = mock_dl.call_args.args[1]
    assert called_dest.name == DEFAULT_METADATA_FILENAME


def test_download_error_propagates(tmp_path: Path, monkeypatch) -> None:
    """Download failures must raise ArtifactDownloadError."""
    _patch_settings(
        monkeypatch,
        artifact_url="http://example.com/model.joblib",
        metadata_url="http://example.com/meta.json",
    )

    with (
        patch.object(
            bootstrap,
            "_download_file",
            side_effect=ArtifactDownloadError("boom"),
        ),
        pytest.raises(ArtifactDownloadError, match="boom"),
    ):
        ensure_model_artifacts(models_dir=tmp_path)
