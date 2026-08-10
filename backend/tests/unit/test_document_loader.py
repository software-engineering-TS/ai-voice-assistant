import pytest

from app.rag.loaders.document_loader import (
    document_loader,
)


def test_extract_text_rejects_empty_path():
    """
    Empty file paths should raise ValueError.
    """

    with pytest.raises(ValueError):

        document_loader.extract_text("")


def test_extract_text_rejects_missing_file(tmp_path):
    """
    Missing files should raise FileNotFoundError.
    """

    file_path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):

        document_loader.extract_text(
            str(file_path)
        )


def test_extract_text_rejects_unsupported_file_type(
    tmp_path,
):
    """
    Unsupported file extensions should raise ValueError.
    """

    file_path = tmp_path / "document.xyz"

    file_path.write_text(
        "Some content.",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):

        document_loader.extract_text(
            str(file_path)
        )


def test_extract_text_uses_txt_loader(tmp_path):
    """
    Verify that TXT files are routed to the TXT loader.
    """

    file_path = tmp_path / "document.txt"

    file_path.write_text(
        "Hello from the document loader.",
        encoding="utf-8",
    )

    result = document_loader.extract_text(
        str(file_path)
    )

    assert result == (
        "Hello from the document loader."
    )


def test_extract_text_supports_pdf(tmp_path):
    """
    Verify that PDF files are routed to the PDF loader.
    """

    import fitz

    file_path = tmp_path / "document.pdf"

    document = fitz.open()

    page = document.new_page()

    page.insert_text(
        (72, 72),
        "PDF document content.",
    )

    document.save(file_path)
    document.close()

    result = document_loader.extract_text(
        str(file_path)
    )

    assert "PDF document content." in result


def test_extract_text_supports_docx(tmp_path):
    """
    Verify that DOCX files are routed to the DOCX loader.
    """

    from docx import Document

    file_path = tmp_path / "document.docx"

    document = Document()

    document.add_paragraph(
        "DOCX document content."
    )

    document.save(file_path)

    result = document_loader.extract_text(
        str(file_path)
    )

    assert "DOCX document content." in result


def test_extract_text_supports_pptx(tmp_path):
    """
    Verify that PPTX files are routed to the PPTX loader.
    """

    from pptx import Presentation

    file_path = tmp_path / "document.pptx"

    presentation = Presentation()

    slide = presentation.slides.add_slide(
        presentation.slide_layouts[6]
    )

    textbox = slide.shapes.add_textbox(
        100,
        100,
        500,
        100,
    )

    textbox.text = "PPTX document content."

    presentation.save(file_path)

    result = document_loader.extract_text(
        str(file_path)
    )

    assert "PPTX document content." in result


def test_extract_text_supports_csv(tmp_path):
    """
    Verify that CSV files are routed to the CSV loader.
    """

    file_path = tmp_path / "document.csv"

    file_path.write_text(
        "Name,Role\nAlice,Engineer\n",
        encoding="utf-8",
    )

    result = document_loader.extract_text(
        str(file_path)
    )

    assert "Columns: Name, Role" in result
    assert "Name: Alice" in result
    assert "Role: Engineer" in result


def test_extract_text_supports_excel(tmp_path):
    """
    Verify that XLSX files are routed to the Excel loader.
    """

    import pandas as pd

    file_path = tmp_path / "document.xlsx"

    dataframe = pd.DataFrame(
        {
            "Name": ["Alice"],
            "Role": ["Engineer"],
        }
    )

    dataframe.to_excel(
        file_path,
        index=False,
        sheet_name="Employees",
    )

    result = document_loader.extract_text(
        str(file_path)
    )

    assert "Sheet: Employees" in result
    assert "Name: Alice" in result
    assert "Role: Engineer" in result


def test_extract_text_rejects_loader_returning_non_string(
    tmp_path,
    monkeypatch,
):
    """
    DocumentLoader should reject an invalid loader result.
    """

    file_path = tmp_path / "document.txt"

    file_path.write_text(
        "Some content.",
        encoding="utf-8",
    )

    class FakeLoader:

        def extract_text(self, file_path):
            return None

    monkeypatch.setitem(
        document_loader.__class__.__module__
        and __import__(
            "app.rag.loaders.document_loader",
            fromlist=["LOADERS"],
        ).LOADERS,
        ".txt",
        FakeLoader(),
    )

    with pytest.raises(ValueError):

        document_loader.extract_text(
            str(file_path)
        )
        
def test_extract_text_rejects_none_path():
    """
    None file paths should raise ValueError.
    """

    with pytest.raises(
        ValueError,
        match="File path cannot be None",
    ):
        document_loader.extract_text(None)
        
def test_extract_text_rejects_empty_path_branch(monkeypatch):
    """
    Explicitly exercise the empty Path validation branch.
    """

    import app.rag.loaders.document_loader as document_module

    class FakePath:
        def __init__(self, value):
            self.value = value

        def __str__(self):
            return ""

    monkeypatch.setattr(
        document_module,
        "Path",
        FakePath,
    )

    with pytest.raises(
        ValueError,
        match="File path cannot be empty",
    ):
        document_loader.extract_text("")
        
def test_extract_text_rejects_file_without_extension(
    tmp_path,
):
    """
    Files without an extension should be rejected.
    """

    file_path = tmp_path / "document"
    file_path.write_text(
        "Some content.",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Document does not have a file extension",
    ):
        document_loader.extract_text(
            file_path
        )
        
def test_extract_text_propagates_loader_failure(
    tmp_path,
    monkeypatch,
):
    """
    Exceptions from an underlying document loader
    should be logged and propagated.
    """

    file_path = tmp_path / "document.txt"

    file_path.write_text(
        "Some content.",
        encoding="utf-8",
    )

    class FailingLoader:

        def extract_text(self, file_path):
            raise RuntimeError(
                "Loader extraction failed"
            )

    import app.rag.loaders.document_loader as document_module

    monkeypatch.setitem(
        document_module.LOADERS,
        ".txt",
        FailingLoader(),
    )

    with pytest.raises(
        RuntimeError,
        match="Loader extraction failed",
    ):
        document_loader.extract_text(
            file_path
        )
        
def test_extract_text_rejects_empty_extracted_text(
    tmp_path,
    monkeypatch,
):
    """
    Empty text returned by a loader should be rejected.
    """

    file_path = tmp_path / "document.txt"

    file_path.write_text(
        "Some content.",
        encoding="utf-8",
    )

    class EmptyLoader:

        def extract_text(self, file_path):
            return "   "

    import app.rag.loaders.document_loader as document_module

    monkeypatch.setitem(
        document_module.LOADERS,
        ".txt",
        EmptyLoader(),
    )

    with pytest.raises(
        ValueError,
        match="No readable text could be extracted",
    ):
        document_loader.extract_text(
            file_path
        )