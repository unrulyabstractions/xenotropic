"""
Tests for download_models.py tool.

Tests for tools/download_models.py
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestIsCached:
    """Tests for is_model_cached function."""

    def test_cached_model_returns_true(self):
        """Test that cached model returns True."""
        from tools.download_models import is_model_cached

        with patch("huggingface_hub.try_to_load_from_cache") as mock:
            mock.return_value = "/path/to/cached/config.json"
            assert is_model_cached("some/model") is True

    def test_uncached_model_returns_false(self):
        """Test that uncached model returns False."""
        from tools.download_models import is_model_cached

        with patch("huggingface_hub.try_to_load_from_cache") as mock:
            mock.return_value = None
            assert is_model_cached("some/model") is False

    def test_exception_returns_false(self):
        """Test that exception returns False."""
        from tools.download_models import is_model_cached

        with patch("huggingface_hub.try_to_load_from_cache") as mock:
            mock.side_effect = Exception("Network error")
            assert is_model_cached("some/model") is False


class TestGetModelSizeGb:
    """Tests for get_model_size_gb function."""

    def test_returns_size_in_gb(self):
        """Test that size is returned in GB."""
        from tools.download_models import get_model_size_gb

        with patch("tools.download_models.HfFileSystem") as mock_fs_cls:
            mock_fs = MagicMock()
            mock_fs.ls.return_value = [
                {"size": 1024**3},  # 1 GB
                {"size": 2 * 1024**3},  # 2 GB
            ]
            mock_fs_cls.return_value = mock_fs

            size = get_model_size_gb("some/model")
            assert size == pytest.approx(3.0)

    def test_exception_returns_none(self):
        """Test that exception returns None."""
        from tools.download_models import get_model_size_gb

        with patch("tools.download_models.HfFileSystem") as mock_fs_cls:
            mock_fs_cls.return_value.ls.side_effect = Exception("Error")
            assert get_model_size_gb("some/model") is None


class TestWarnIfNotCached:
    """Tests for warn_if_not_cached function."""

    def test_cached_model_no_warning(self, capsys):
        """Test that cached model produces no warning."""
        from tools.download_models import warn_if_not_cached

        with patch("tools.download_models.is_model_cached", return_value=True):
            result = warn_if_not_cached("cached/model")

        assert result is True
        captured = capsys.readouterr()
        assert "WARNING" not in captured.out

    def test_uncached_model_prints_warning(self, capsys):
        """Test that uncached model prints warning."""
        from tools.download_models import warn_if_not_cached

        with patch("tools.download_models.is_model_cached", return_value=False):
            with patch("tools.download_models.get_model_size_gb", return_value=5.0):
                result = warn_if_not_cached("uncached/model")

        assert result is False
        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "uncached/model" in captured.out
        assert "5.0 GB" in captured.out


class TestDownloadModelsInput:
    """Tests for DownloadModelsInput dataclass."""

    def test_from_json(self, tmp_path):
        """Test loading from JSON file."""
        from tools.download_models import DownloadModelsInput

        config_path = tmp_path / "config.json"
        config_path.write_text('{"models": ["model1", "model2"]}')

        inp = DownloadModelsInput.from_json(config_path)
        assert inp.models == ["model1", "model2"]
        assert inp.skip_validation is False

    def test_from_json_with_skip_validation(self, tmp_path):
        """Test loading with skip_validation option."""
        from tools.download_models import DownloadModelsInput

        config_path = tmp_path / "config.json"
        config_path.write_text('{"models": ["model1"], "skip_validation": true}')

        inp = DownloadModelsInput.from_json(config_path)
        assert inp.skip_validation is True


class TestModelDownloadResult:
    """Tests for ModelDownloadResult dataclass."""

    def test_basic_creation(self):
        """Test creating a result."""
        from tools.download_models import ModelDownloadResult

        result = ModelDownloadResult(
            model_name="test/model",
            success=True,
            cached=False,
            download_time=10.5,
        )
        assert result.model_name == "test/model"
        assert result.success is True
        assert result.error is None


class TestDownloadModelsOutput:
    """Tests for DownloadModelsOutput dataclass."""

    def test_to_dict(self):
        """Test conversion to dict."""
        from tools.download_models import DownloadModelsOutput, ModelDownloadResult

        output = DownloadModelsOutput(
            results=[
                ModelDownloadResult("m1", True, True, 0.0),
                ModelDownloadResult("m2", True, False, 5.0),
                ModelDownloadResult("m3", False, False, 1.0, "Error"),
            ]
        )

        d = output.to_dict()
        assert d["summary"]["total"] == 3
        assert d["summary"]["success"] == 2
        assert d["summary"]["failed"] == 1
        assert d["summary"]["already_cached"] == 1


class TestDownloadModel:
    """Tests for download_model function."""

    def test_cached_model_skips_download(self, capsys):
        """Test that cached model skips download."""
        from tools.download_models import download_model

        with patch("tools.download_models.is_model_cached", return_value=True):
            result = download_model("cached/model")

        assert result.success is True
        assert result.cached is True
        assert result.download_time == 0.0
        captured = capsys.readouterr()
        assert "[CACHED]" in captured.out

    def test_download_failure_returns_error(self, capsys):
        """Test that download failure returns error."""
        from tools.download_models import download_model

        with patch("tools.download_models.is_model_cached", return_value=False):
            with patch("tools.download_models.get_model_size_gb", return_value=1.0):
                with patch(
                    "tools.download_models.AutoTokenizer.from_pretrained"
                ) as mock:
                    mock.side_effect = Exception("Download failed")
                    result = download_model("failing/model")

        assert result.success is False
        assert result.error is not None
        assert "Download failed" in result.error

    def test_gated_model_helpful_error(self, capsys):
        """Test that gated model gives helpful error."""
        from tools.download_models import download_model

        with patch("tools.download_models.is_model_cached", return_value=False):
            with patch("tools.download_models.get_model_size_gb", return_value=1.0):
                with patch(
                    "tools.download_models.AutoTokenizer.from_pretrained"
                ) as mock:
                    mock.side_effect = Exception("403 gated repo")
                    result = download_model("meta-llama/Llama-3.1-8B")

        assert result.success is False
        assert "huggingface-cli login" in result.error


class TestCheckCacheOnly:
    """Tests for check_cache_only function."""

    def test_all_cached_returns_zero(self, capsys):
        """Test that all cached returns 0."""
        from tools.download_models import DownloadModelsInput, check_cache_only

        inp = DownloadModelsInput(models=["m1", "m2"])

        with patch("tools.download_models.is_model_cached", return_value=True):
            result = check_cache_only(inp)

        assert result == 0
        captured = capsys.readouterr()
        assert "Cached: 2/2" in captured.out

    def test_some_uncached_returns_one(self, capsys):
        """Test that some uncached returns 1."""
        from tools.download_models import DownloadModelsInput, check_cache_only

        inp = DownloadModelsInput(models=["m1", "m2"])

        def mock_cached(name):
            return name == "m1"

        with patch("tools.download_models.is_model_cached", side_effect=mock_cached):
            with patch("tools.download_models.get_model_size_gb", return_value=1.0):
                result = check_cache_only(inp)

        assert result == 1
        captured = capsys.readouterr()
        assert "Cached: 1/2" in captured.out


class TestGetCachedModelSizeGb:
    """Tests for get_cached_model_size_gb function."""

    def test_returns_size_for_cached_model(self):
        """Test that size is returned for cached model."""
        from tools.download_models import get_cached_model_size_gb

        mock_repo = MagicMock()
        mock_repo.repo_id = "test/model"
        mock_repo.size_on_disk = 5 * 1024**3  # 5 GB

        mock_cache = MagicMock()
        mock_cache.repos = [mock_repo]

        with patch("huggingface_hub.scan_cache_dir", return_value=mock_cache):
            size = get_cached_model_size_gb("test/model")
            assert size == pytest.approx(5.0)

    def test_returns_none_for_uncached_model(self):
        """Test that None is returned for uncached model."""
        from tools.download_models import get_cached_model_size_gb

        mock_cache = MagicMock()
        mock_cache.repos = []

        with patch("huggingface_hub.scan_cache_dir", return_value=mock_cache):
            size = get_cached_model_size_gb("nonexistent/model")
            assert size is None

    def test_exception_returns_none(self):
        """Test that exception returns None."""
        from tools.download_models import get_cached_model_size_gb

        with patch("huggingface_hub.scan_cache_dir", side_effect=Exception("Error")):
            assert get_cached_model_size_gb("any/model") is None


class TestListAllCachedModels:
    """Tests for list_all_cached_models function."""

    def test_returns_sorted_models(self):
        """Test that models are sorted by size descending."""
        from tools.download_models import list_all_cached_models

        mock_repos = []
        for name, size in [("small", 1), ("large", 10), ("medium", 5)]:
            repo = MagicMock()
            repo.repo_id = name
            repo.repo_type = "model"
            repo.size_on_disk = size * 1024**3
            mock_repos.append(repo)

        mock_cache = MagicMock()
        mock_cache.repos = mock_repos

        with patch("huggingface_hub.scan_cache_dir", return_value=mock_cache):
            models = list_all_cached_models()

        assert len(models) == 3
        assert models[0][0] == "large"
        assert models[1][0] == "medium"
        assert models[2][0] == "small"

    def test_filters_to_model_type_only(self):
        """Test that only model repos are returned."""
        from tools.download_models import list_all_cached_models

        repos = []
        for name, repo_type in [("model1", "model"), ("dataset1", "dataset")]:
            repo = MagicMock()
            repo.repo_id = name
            repo.repo_type = repo_type
            repo.size_on_disk = 1024**3
            repos.append(repo)

        mock_cache = MagicMock()
        mock_cache.repos = repos

        with patch("huggingface_hub.scan_cache_dir", return_value=mock_cache):
            models = list_all_cached_models()

        assert len(models) == 1
        assert models[0][0] == "model1"


class TestPrintCachedModelsReport:
    """Tests for print_cached_models_report function."""

    def test_prints_models_with_box_formatting(self, capsys):
        """Test that report has box formatting."""
        from tools.download_models import print_cached_models_report

        with patch(
            "tools.download_models.list_all_cached_models",
            return_value=[("test/model", 5.0)],
        ):
            print_cached_models_report()

        captured = capsys.readouterr()
        assert "CACHED MODELS" in captured.out
        assert "test/model" in captured.out
        assert "5.00 GB" in captured.out
        assert "TOTAL" in captured.out
        # Check box characters
        assert "╔" in captured.out
        assert "╚" in captured.out

    def test_prints_no_models_message(self, capsys):
        """Test that empty cache shows message."""
        from tools.download_models import print_cached_models_report

        with patch("tools.download_models.list_all_cached_models", return_value=[]):
            print_cached_models_report()

        captured = capsys.readouterr()
        assert "No models cached" in captured.out
