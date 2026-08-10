import pytest

from app.rag.text_splitter import (
    text_splitter_service,
)


def test_split_creates_chunks():
    """
    Verify that normal document text is split
    into multiple chunks.
    """

    text = (
        "This is a test document. "
        * 200
    )

    chunks = text_splitter_service.split(
        text
    )

    assert isinstance(chunks, list)
    assert len(chunks) > 1

    assert all(
        isinstance(chunk, str)
        for chunk in chunks
    )

    assert all(
        chunk.strip()
        for chunk in chunks
    )


def test_split_returns_empty_list_for_empty_text():
    """
    Empty document text should return an empty list.
    """

    result = text_splitter_service.split("")

    assert result == []


def test_split_returns_empty_list_for_whitespace():
    """
    Whitespace-only document text should return
    an empty list.
    """

    result = text_splitter_service.split(
        "   \n\n   "
    )

    assert result == []


def test_split_rejects_non_string_input():
    """
    Document text must be a string.
    """

    with pytest.raises(TypeError):

        text_splitter_service.split(
            None
        )


def test_split_removes_empty_chunks():
    """
    Returned chunks must contain actual content.
    """

    text = (
        "First section.\n\n"
        "Second section.\n\n"
        "Third section."
    )

    chunks = text_splitter_service.split(
        text
    )

    assert chunks

    for chunk in chunks:
        assert chunk.strip() == chunk
        assert chunk.strip()


def test_split_preserves_complete_document_content():
    """
    Verify that all meaningful words from the original
    document remain represented in the generated chunks.
    """

    text = (
        "Python is used for backend development. "
        "FastAPI is used to build APIs. "
        "ChromaDB is used for vector storage."
    )

    chunks = text_splitter_service.split(
        text
    )

    combined_text = " ".join(chunks)

    assert "Python" in combined_text
    assert "FastAPI" in combined_text
    assert "ChromaDB" in combined_text


def test_chunk_configuration_is_valid():
    """
    Verify that the configured chunk size and overlap
    satisfy the expected relationship.
    """

    assert (
        text_splitter_service.chunk_size > 0
    )

    assert (
        text_splitter_service.chunk_overlap >= 0
    )

    assert (
        text_splitter_service.chunk_overlap
        < text_splitter_service.chunk_size
    )
    
import app.rag.text_splitter as splitter_module


def test_service_rejects_zero_chunk_size(monkeypatch):
    """
    CHUNK_SIZE must be greater than zero.
    """

    monkeypatch.setattr(
        splitter_module.settings,
        "CHUNK_SIZE",
        0,
    )

    monkeypatch.setattr(
        splitter_module.settings,
        "CHUNK_OVERLAP",
        10,
    )

    with pytest.raises(
        ValueError,
        match="CHUNK_SIZE must be greater than zero",
    ):
        splitter_module.TextSplitterService()


def test_service_rejects_negative_chunk_overlap(monkeypatch):
    """
    CHUNK_OVERLAP cannot be negative.
    """

    monkeypatch.setattr(
        splitter_module.settings,
        "CHUNK_SIZE",
        100,
    )

    monkeypatch.setattr(
        splitter_module.settings,
        "CHUNK_OVERLAP",
        -1,
    )

    with pytest.raises(
        ValueError,
        match="CHUNK_OVERLAP cannot be negative",
    ):
        splitter_module.TextSplitterService()


def test_service_rejects_overlap_equal_to_chunk_size(
    monkeypatch,
):
    """
    CHUNK_OVERLAP must be smaller than CHUNK_SIZE.
    """

    monkeypatch.setattr(
        splitter_module.settings,
        "CHUNK_SIZE",
        100,
    )

    monkeypatch.setattr(
        splitter_module.settings,
        "CHUNK_OVERLAP",
        100,
    )

    with pytest.raises(
        ValueError,
        match="CHUNK_OVERLAP must be smaller",
    ):
        splitter_module.TextSplitterService()


def test_split_propagates_splitter_failure(
    monkeypatch,
):
    """
    Unexpected splitter errors should be propagated.
    """

    def fail_split(text):
        raise RuntimeError(
            "Splitting failed"
        )

    monkeypatch.setattr(
        text_splitter_service.splitter,
        "split_text",
        fail_split,
    )

    with pytest.raises(
        RuntimeError,
        match="Splitting failed",
    ):
        text_splitter_service.split(
            "Some valid document text."
        )


def test_split_skips_empty_chunks(
    monkeypatch,
):
    """
    Empty chunks should be ignored.
    """

    monkeypatch.setattr(
        text_splitter_service.splitter,
        "split_text",
        lambda text: [
            "",
            "Valid chunk",
        ],
    )

    result = text_splitter_service.split(
        "Some valid text"
    )

    assert result == [
        "Valid chunk"
    ]


def test_split_skips_whitespace_chunks(
    monkeypatch,
):
    """
    Whitespace-only chunks should be ignored.
    """

    monkeypatch.setattr(
        text_splitter_service.splitter,
        "split_text",
        lambda text: [
            "   ",
            "\n\n",
            "Valid chunk",
        ],
    )

    result = text_splitter_service.split(
        "Some valid text"
    )

    assert result == [
        "Valid chunk"
    ]


def test_split_returns_empty_when_no_usable_chunks(
    monkeypatch,
):
    """
    If the splitter produces no usable chunks,
    an empty list should be returned.
    """

    monkeypatch.setattr(
        text_splitter_service.splitter,
        "split_text",
        lambda text: [
            "",
            "   ",
            "\n\n",
        ],
    )

    result = text_splitter_service.split(
        "Some valid document text."
    )

    assert result == []