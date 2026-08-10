import hashlib
import pytest

from app.rag.utils.file_hash import file_hash


def test_sha256_returns_correct_hash(tmp_path):
    """
    Verify that SHA-256 is calculated correctly.
    """

    file_path = tmp_path / "test.txt"

    content = "Hello World"

    file_path.write_text(
        content,
        encoding="utf-8",
    )

    expected_hash = hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()

    result = file_hash.sha256(
        str(file_path)
    )

    assert result == expected_hash


def test_sha256_returns_64_character_hash(tmp_path):
    """
    SHA-256 hexadecimal digest must contain
    exactly 64 characters.
    """

    file_path = tmp_path / "test.txt"

    file_path.write_text(
        "Test document",
        encoding="utf-8",
    )

    result = file_hash.sha256(
        str(file_path)
    )

    assert len(result) == 64


def test_same_content_produces_same_hash(tmp_path):
    """
    Identical file contents must produce
    identical hashes.
    """

    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"

    content = "Same content"

    file1.write_text(
        content,
        encoding="utf-8",
    )

    file2.write_text(
        content,
        encoding="utf-8",
    )

    hash1 = file_hash.sha256(
        str(file1)
    )

    hash2 = file_hash.sha256(
        str(file2)
    )

    assert hash1 == hash2


def test_different_content_produces_different_hash(
    tmp_path,
):
    """
    Different file contents must produce
    different hashes.
    """

    file1 = tmp_path / "file1.txt"
    file2 = tmp_path / "file2.txt"

    file1.write_text(
        "Content A",
        encoding="utf-8",
    )

    file2.write_text(
        "Content B",
        encoding="utf-8",
    )

    hash1 = file_hash.sha256(
        str(file1)
    )

    hash2 = file_hash.sha256(
        str(file2)
    )

    assert hash1 != hash2
    
def test_sha256_rejects_none_path():
    """
    None file paths should raise ValueError.
    """

    with pytest.raises(
        ValueError,
        match="File path cannot be None",
    ):
        file_hash.sha256(None)


def test_sha256_rejects_missing_file(tmp_path):
    """
    Missing files should raise FileNotFoundError.
    """

    file_path = tmp_path / "missing.txt"

    with pytest.raises(
        FileNotFoundError,
        match="File not found",
    ):
        file_hash.sha256(file_path)


def test_sha256_rejects_directory(tmp_path):
    """
    Directory paths should raise ValueError.
    """

    directory = tmp_path / "test_directory"
    directory.mkdir()

    with pytest.raises(
        ValueError,
        match="Path is not a file",
    ):
        file_hash.sha256(directory)


def test_sha256_propagates_os_error(
    monkeypatch,
    tmp_path,
):
    """
    OSError while opening the file should be
    logged and propagated.
    """

    file_path = tmp_path / "test.txt"

    file_path.write_text(
        "Test content",
        encoding="utf-8",
    )

    def raise_os_error(
        self,
        *args,
        **kwargs,
    ):
        raise OSError(
            "Unable to open file"
        )

    monkeypatch.setattr(
        type(file_path),
        "open",
        raise_os_error,
    )

    with pytest.raises(
        OSError,
        match="Unable to open file",
    ):
        file_hash.sha256(file_path)