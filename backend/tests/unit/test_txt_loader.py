import pytest
from pathlib import Path

from app.rag.loaders.txt_loader import txt_loader


def test_extract_text_successfully(tmp_path):
    """
    Verify that a normal TXT file is read correctly.
    """

    file_path = tmp_path / "test.txt"

    file_path.write_text(
        "Hello from the AI Voice Assistant.",
        encoding="utf-8",
    )

    result = txt_loader.extract_text(
        str(file_path)
    )

    assert result == (
        "Hello from the AI Voice Assistant."
    )


def test_extract_text_strips_outer_whitespace(tmp_path):
    """
    Leading and trailing whitespace should be removed.
    """

    file_path = tmp_path / "test.txt"

    file_path.write_text(
        "   Hello World   \n\n",
        encoding="utf-8",
    )

    result = txt_loader.extract_text(
        str(file_path)
    )

    assert result == "Hello World"


def test_extract_text_rejects_empty_file(tmp_path):
    """
    An empty TXT file should raise ValueError.
    """

    file_path = tmp_path / "empty.txt"

    file_path.write_text(
        "",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):

        txt_loader.extract_text(
            str(file_path)
        )


def test_extract_text_rejects_whitespace_only_file(
    tmp_path,
):
    """
    A whitespace-only TXT file should raise ValueError.
    """

    file_path = tmp_path / "empty.txt"

    file_path.write_text(
        "   \n\n   ",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):

        txt_loader.extract_text(
            str(file_path)
        )


def test_extract_text_rejects_empty_path():
    """
    Empty file paths should raise ValueError.
    """

    with pytest.raises(ValueError):

        txt_loader.extract_text("")


def test_extract_text_rejects_missing_file(tmp_path):
    """
    A nonexistent file should raise FileNotFoundError.
    """

    file_path = tmp_path / "does_not_exist.txt"

    with pytest.raises(FileNotFoundError):

        txt_loader.extract_text(
            str(file_path)
        )


def test_extract_text_rejects_invalid_utf8(tmp_path):
    """
    Invalid UTF-8 content should raise ValueError.
    """

    file_path = tmp_path / "invalid.txt"

    file_path.write_bytes(
        b"\xff\xfe\xfd"
    )

    with pytest.raises(ValueError):

        txt_loader.extract_text(
            str(file_path)
        )
        
def test_extract_text_rejects_none_path():
    """
    None paths should raise ValueError.
    """

    with pytest.raises(
        ValueError,
        match="TXT path cannot be None",
    ):
        txt_loader.extract_text(None)


def test_extract_text_rejects_directory(tmp_path):
    """
    Directory paths should raise ValueError.
    """

    directory = tmp_path / "txt_directory"
    directory.mkdir()

    with pytest.raises(
        ValueError,
        match="TXT path is not a file",
    ):
        txt_loader.extract_text(directory)


def test_extract_text_rejects_non_txt_file(tmp_path):
    """
    Files with non-TXT extensions should be rejected.
    """

    file_path = tmp_path / "test.csv"

    file_path.write_text(
        "Name,Role\nAlice,Engineer\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Provided file is not a TXT file",
    ):
        txt_loader.extract_text(file_path)


def test_extract_text_handles_read_os_error(
    monkeypatch,
    tmp_path,
):
    """
    OSError while reading the TXT file should be
    converted to ValueError.
    """

    file_path = tmp_path / "test.txt"

    file_path.write_text(
        "Hello World",
        encoding="utf-8",
    )

    def raise_os_error(
        self,
        *args,
        **kwargs,
    ):
        raise OSError(
            "Unable to read TXT file"
        )

    monkeypatch.setattr(
        Path,
        "read_text",
        raise_os_error,
    )

    with pytest.raises(
        ValueError,
        match="Unable to read the TXT file",
    ):
        txt_loader.extract_text(file_path)
        
def test_extract_text_rejects_empty_path(
    monkeypatch,
):
    """
    Empty paths should raise the specific
    empty-path validation error.
    """

    class FakePath:
        def __init__(self, value):
            self.value = value

        def __str__(self):
            return ""

    monkeypatch.setattr(
        "app.rag.loaders.txt_loader.Path",
        FakePath,
    )

    with pytest.raises(
        ValueError,
        match="TXT path cannot be empty",
    ):
        txt_loader.extract_text("")