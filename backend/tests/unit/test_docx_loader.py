import pytest
from docx import Document

from app.rag.loaders.docx_loader import docx_loader


def test_extract_text_from_paragraphs(tmp_path):
    """
    Verify that DOCX paragraphs are extracted correctly.
    """

    file_path = tmp_path / "test.docx"

    document = Document()

    document.add_paragraph(
        "First paragraph."
    )

    document.add_paragraph(
        "Second paragraph."
    )

    document.save(file_path)

    result = docx_loader.extract_text(
        str(file_path)
    )

    assert "First paragraph." in result
    assert "Second paragraph." in result


def test_extract_text_from_tables(tmp_path):
    """
    Verify that DOCX table cells are extracted.
    """

    file_path = tmp_path / "table.docx"

    document = Document()

    table = document.add_table(
        rows=2,
        cols=2,
    )

    table.cell(0, 0).text = "Name"
    table.cell(0, 1).text = "Role"

    table.cell(1, 0).text = "Alice"
    table.cell(1, 1).text = "Engineer"

    document.save(file_path)

    result = docx_loader.extract_text(
        str(file_path)
    )

    assert "Name | Role" in result
    assert "Alice | Engineer" in result


def test_extract_text_ignores_empty_paragraphs(tmp_path):
    """
    Empty paragraphs should not appear in the output.
    """

    file_path = tmp_path / "test.docx"

    document = Document()

    document.add_paragraph("")
    document.add_paragraph("   ")
    document.add_paragraph("Valid paragraph.")

    document.save(file_path)

    result = docx_loader.extract_text(
        str(file_path)
    )

    assert result == "Valid paragraph."


def test_extract_text_strips_whitespace(tmp_path):
    """
    Paragraph whitespace should be stripped.
    """

    file_path = tmp_path / "test.docx"

    document = Document()

    document.add_paragraph(
        "   Hello World   "
    )

    document.save(file_path)

    result = docx_loader.extract_text(
        str(file_path)
    )

    assert result == "Hello World"


def test_extract_text_rejects_empty_path():
    """
    Empty paths should raise ValueError.
    """

    with pytest.raises(ValueError):

        docx_loader.extract_text("")


def test_extract_text_rejects_missing_file(tmp_path):
    """
    Missing DOCX files should raise FileNotFoundError.
    """

    file_path = (
        tmp_path / "missing.docx"
    )

    with pytest.raises(FileNotFoundError):

        docx_loader.extract_text(
            str(file_path)
        )


def test_extract_text_rejects_empty_document(
    tmp_path,
):
    """
    A DOCX containing no readable text should
    raise ValueError.
    """

    file_path = tmp_path / "empty.docx"

    document = Document()

    document.save(file_path)

    with pytest.raises(ValueError):

        docx_loader.extract_text(
            str(file_path)
        )


def test_extract_text_rejects_invalid_docx(
    tmp_path,
):
    """
    A corrupted/invalid DOCX file should raise ValueError.
    """

    file_path = tmp_path / "invalid.docx"

    file_path.write_bytes(
        b"This is not a valid DOCX file."
    )

    with pytest.raises(ValueError):

        docx_loader.extract_text(
            str(file_path)
        )
        
def test_extract_text_rejects_none_path():
    """
    None DOCX paths should raise ValueError.
    """

    with pytest.raises(
        ValueError,
        match="DOCX path cannot be None",
    ):
        docx_loader.extract_text(None)


def test_extract_text_rejects_empty_path_with_message():
    """
    Empty DOCX paths should raise the expected ValueError.
    """

    with pytest.raises(
        ValueError,
        match="DOCX path cannot be empty",
    ):
        docx_loader.extract_text("")


def test_extract_text_rejects_non_docx_file(tmp_path):
    """
    Files with extensions other than .docx should be rejected.
    """

    file_path = tmp_path / "test.txt"

    file_path.write_text(
        "This is not a DOCX file.",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Provided file is not a DOCX file",
    ):
        docx_loader.extract_text(file_path)
        
def test_extract_text_rejects_directory_path(tmp_path):
    """
    A directory path should raise ValueError.
    """

    directory = tmp_path / "documents"
    directory.mkdir()

    with pytest.raises(
        ValueError,
        match="DOCX path is not a file",
    ):
        docx_loader.extract_text(directory)