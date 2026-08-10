import pytest

from app.rag.metadata_builder import (
    metadata_builder,
)


def test_build_creates_metadata_for_each_chunk():
    """
    Verify that one metadata record is created
    for every chunk.
    """

    metadata = metadata_builder.build(
        file_path="uploads/resume.pdf",
        total_chunks=3,
        file_hash="a" * 64,
    )

    assert isinstance(metadata, list)
    assert len(metadata) == 3


def test_metadata_contains_correct_file_information():
    """
    Verify source and file type metadata.
    """

    metadata = metadata_builder.build(
        file_path="uploads/resume.pdf",
        total_chunks=2,
        file_hash="a" * 64,
    )

    assert metadata[0]["source"] == "resume.pdf"
    assert metadata[0]["file_type"] == "pdf"


def test_metadata_contains_correct_chunk_numbers():
    """
    Chunk numbering should start at 1 and increment
    sequentially.
    """

    metadata = metadata_builder.build(
        file_path="uploads/resume.pdf",
        total_chunks=3,
        file_hash="a" * 64,
    )

    assert metadata[0]["chunk"] == 1
    assert metadata[1]["chunk"] == 2
    assert metadata[2]["chunk"] == 3


def test_metadata_contains_file_hash():
    """
    Every chunk must contain the document hash.
    """

    file_hash = "b" * 64

    metadata = metadata_builder.build(
        file_path="uploads/resume.pdf",
        total_chunks=3,
        file_hash=file_hash,
    )

    for item in metadata:
        assert item["file_hash"] == file_hash


def test_build_rejects_non_string_file_path():
    """
    file_path must be a string.
    """

    with pytest.raises(TypeError):

        metadata_builder.build(
            file_path=None,
            total_chunks=3,
            file_hash="a" * 64,
        )


def test_build_rejects_empty_file_path():
    """
    Empty file paths are invalid.
    """

    with pytest.raises(ValueError):

        metadata_builder.build(
            file_path="",
            total_chunks=3,
            file_hash="a" * 64,
        )


def test_build_rejects_invalid_chunk_count():
    """
    total_chunks must be a positive integer.
    """

    with pytest.raises(ValueError):

        metadata_builder.build(
            file_path="uploads/resume.pdf",
            total_chunks=0,
            file_hash="a" * 64,
        )


def test_build_rejects_invalid_hash():
    """
    Empty hashes are invalid.
    """

    with pytest.raises(ValueError):

        metadata_builder.build(
            file_path="uploads/resume.pdf",
            total_chunks=3,
            file_hash="",
        )


def test_metadata_has_expected_structure():
    """
    Verify that every metadata record contains
    all required fields.
    """

    metadata = metadata_builder.build(
        file_path="uploads/report.docx",
        total_chunks=2,
        file_hash="c" * 64,
    )

    required_fields = {
        "source",
        "file_type",
        "chunk",
        "file_hash",
    }

    for item in metadata:
        assert set(item.keys()) == required_fields
        
def test_build_rejects_non_integer_chunk_count():
    """
    total_chunks must be an integer.
    """

    with pytest.raises(
        TypeError,
        match="total_chunks must be an integer",
    ):
        metadata_builder.build(
            file_path="uploads/resume.pdf",
            total_chunks="3",
            file_hash="a" * 64,
        )


def test_build_rejects_boolean_chunk_count():
    """
    Boolean values must not be accepted as chunk counts.
    """

    with pytest.raises(
        TypeError,
        match="total_chunks must be an integer",
    ):
        metadata_builder.build(
            file_path="uploads/resume.pdf",
            total_chunks=True,
            file_hash="a" * 64,
        )


def test_build_rejects_non_string_file_hash():
    """
    file_hash must be a string.
    """

    with pytest.raises(
        TypeError,
        match="file_hash must be a string",
    ):
        metadata_builder.build(
            file_path="uploads/resume.pdf",
            total_chunks=3,
            file_hash=None,
        )


def test_build_rejects_invalid_sha256_hash():
    """
    file_hash must contain a valid SHA-256 hash.
    """

    with pytest.raises(
        ValueError,
        match="file_hash must be a valid SHA-256 hash",
    ):
        metadata_builder.build(
            file_path="uploads/resume.pdf",
            total_chunks=3,
            file_hash="invalid-hash",
        )


def test_build_rejects_file_without_extension():
    """
    Files without an extension should be rejected.
    """

    with pytest.raises(
        ValueError,
        match="Unable to determine file type",
    ):
        metadata_builder.build(
            file_path="uploads/resume",
            total_chunks=3,
            file_hash="a" * 64,
        )


def test_build_rejects_unable_to_determine_file_name(
    monkeypatch,
):
    """
    A path that produces no filename should be rejected.
    """

    import app.rag.metadata_builder as metadata_module

    class FakePath:
        def __init__(self, value):
            self.value = value
            self.name = ""
            self.suffix = ".pdf"

    monkeypatch.setattr(
        metadata_module,
        "Path",
        FakePath,
    )

    with pytest.raises(
        ValueError,
        match="Unable to determine file name",
    ):
        metadata_builder.build(
            file_path="uploads/resume.pdf",
            total_chunks=3,
            file_hash="a" * 64,
        )