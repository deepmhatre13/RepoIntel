import pytest

from packages.ingestion.exceptions import InvalidRepositoryError
from packages.ingestion.security import validate_local_repo_path, validate_repo_url


def test_http_url_validator_rejects_local_directory(tmp_path):
    with pytest.raises(InvalidRepositoryError):
        validate_repo_url(str(tmp_path))


def test_explicit_local_path_validator_accepts_path_under_root(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()

    parsed = validate_local_repo_path(str(repository), str(tmp_path))

    assert parsed.scheme == "local"
    assert parsed.path == str(repository.resolve())


def test_explicit_local_path_validator_rejects_paths_outside_root(tmp_path):
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()

    with pytest.raises(InvalidRepositoryError, match="within the allowed root"):
        validate_local_repo_path(str(outside), str(allowed_root))