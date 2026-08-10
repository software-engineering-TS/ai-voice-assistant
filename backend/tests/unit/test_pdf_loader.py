import fitz
import pytest

from app.rag.loaders.pdf_loader import pdf_loader
import app.rag.loaders.pdf_loader as pdf_module


def test_extract_text_successfully(tmp_path):
    """
    Verify that text is extracted from a PDF.
    """

    file_path = tmp_path / "test.pdf"

    document = fitz.open()

    page = document.new_page()

    page.insert_text(
        (72, 72),
        "Hello from the PDF document.",
    )

    document.save(file_path)
    document.close()

    result = pdf_loader.extract_text(
        str(file_path)
    )

    assert "Hello from the PDF document." in result


def test_extract_text_from_multiple_pages(tmp_path):
    """
    Verify that text from multiple PDF pages
    is extracted.
    """

    file_path = tmp_path / "multi_page.pdf"

    document = fitz.open()

    page1 = document.new_page()

    page1.insert_text(
        (72, 72),
        "First page content.",
    )

    page2 = document.new_page()

    page2.insert_text(
        (72, 72),
        "Second page content.",
    )

    document.save(file_path)
    document.close()

    result = pdf_loader.extract_text(
        str(file_path)
    )

    assert "First page content." in result
    assert "Second page content." in result


def test_extract_text_preserves_page_boundaries(
    tmp_path,
):
    """
    Text from separate pages should remain separated.
    """

    file_path = tmp_path / "pages.pdf"

    document = fitz.open()

    page1 = document.new_page()

    page1.insert_text(
        (72, 72),
        "Page One",
    )

    page2 = document.new_page()

    page2.insert_text(
        (72, 72),
        "Page Two",
    )

    document.save(file_path)
    document.close()

    result = pdf_loader.extract_text(
        str(file_path)
    )

    assert "Page One\n\nPage Two" in result


def test_extract_text_rejects_empty_path(monkeypatch):
    class FakePath:
        def __init__(self, value):
            self.value = value

        def __str__(self):
            return ""

    monkeypatch.setattr(
        pdf_module,
        "Path",
        FakePath,
    )

    with pytest.raises(
        ValueError,
        match="PDF path cannot be empty",
    ):
        pdf_loader.extract_text("")


def test_extract_text_rejects_missing_file(tmp_path):
    """
    Missing PDF files should raise FileNotFoundError.
    """

    file_path = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):

        pdf_loader.extract_text(
            str(file_path)
        )


def test_extract_text_rejects_empty_pdf(tmp_path):
    """
    A PDF containing no readable text should
    raise ValueError.
    """

    file_path = tmp_path / "empty.pdf"

    document = fitz.open()

    document.new_page()

    document.save(file_path)
    document.close()

    with pytest.raises(ValueError):

        pdf_loader.extract_text(
            str(file_path)
        )


def test_extract_text_rejects_invalid_pdf(tmp_path):
    """
    Invalid PDF content should raise ValueError.
    """

    file_path = tmp_path / "invalid.pdf"

    file_path.write_bytes(
        b"This is not a valid PDF file."
    )

    with pytest.raises(ValueError):

        pdf_loader.extract_text(
            str(file_path)
        )


def test_extract_text_strips_page_whitespace(tmp_path):
    """
    Extracted page text should not contain unnecessary
    leading or trailing whitespace.
    """

    file_path = tmp_path / "whitespace.pdf"

    document = fitz.open()

    page = document.new_page()

    page.insert_text(
        (72, 72),
        "Clean PDF text",
    )

    document.save(file_path)
    document.close()

    result = pdf_loader.extract_text(
        str(file_path)
    )

    assert result == result.strip()
    
def test_extract_text_rejects_none_path():
    with pytest.raises(
        ValueError,
        match="PDF path cannot be None",
    ):
        pdf_loader.extract_text(None)
        
def test_extract_text_rejects_empty_path():
    with pytest.raises(
        ValueError,
        match="PDF path cannot be empty",
    ):
        pdf_loader.extract_text("")
        
def test_extract_text_rejects_non_pdf_file(tmp_path):
    """
    Non-PDF files should be rejected.
    """

    file_path = tmp_path / "test.txt"

    file_path.write_text(
        "This is not a PDF.",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Provided file is not a PDF",
    ):
        pdf_loader.extract_text(
            file_path
        )
        
def test_extract_text_rejects_pdf_with_no_pages(
    monkeypatch,
    tmp_path,
):
    file_path = tmp_path / "empty.pdf"
    file_path.write_bytes(b"fake pdf")

    class EmptyDocument:
        def __len__(self):
            return 0

        def __enter__(self):
            return self

        def __exit__(
            self,
            exc_type,
            exc_value,
            traceback,
        ):
            pass

    monkeypatch.setattr(
        pdf_module.fitz,
        "open",
        lambda *args, **kwargs: EmptyDocument(),
    )

    with pytest.raises(
        ValueError,
        match="PDF contains no pages",
    ):
        pdf_loader.extract_text(file_path)
        
def test_extract_text_skips_page_when_extraction_fails(
    monkeypatch,
    tmp_path,
):
    file_path = tmp_path / "test.pdf"
    file_path.write_bytes(b"fake pdf")

    class FailingPage:
        def get_text(self, mode):
            raise RuntimeError(
                "Page extraction failed"
            )

    class FakeDocument:
        def __len__(self):
            return 1

        def __iter__(self):
            return iter(
                [FailingPage()]
            )

        def __enter__(self):
            return self

        def __exit__(
            self,
            exc_type,
            exc_value,
            traceback,
        ):
            pass

    monkeypatch.setattr(
        pdf_module.fitz,
        "open",
        lambda *args, **kwargs: FakeDocument(),
    )

    with pytest.raises(
        ValueError,
        match="No readable text could be extracted",
    ):
        pdf_loader.extract_text(file_path)
        
def test_extract_text_rejects_corrupted_pdf(
    monkeypatch,
    tmp_path,
):
    file_path = tmp_path / "corrupted.pdf"
    file_path.write_bytes(b"corrupted")

    def raise_file_data_error(*args, **kwargs):
        raise fitz.FileDataError(
            "Invalid PDF"
        )

    monkeypatch.setattr(
        pdf_module.fitz,
        "open",
        raise_file_data_error,
    )

    with pytest.raises(
        ValueError,
        match="The PDF file is invalid or corrupted",
    ):
        pdf_loader.extract_text(file_path)
        
def test_extract_text_rejects_pdf_with_no_readable_text(
    monkeypatch,
    tmp_path,
):
    file_path = tmp_path / "image_based.pdf"
    file_path.write_bytes(b"fake pdf")

    class EmptyTextPage:
        def get_text(self, mode):
            return "   "

    class FakeDocument:
        def __len__(self):
            return 1

        def __iter__(self):
            return iter(
                [EmptyTextPage()]
            )

        def __enter__(self):
            return self

        def __exit__(
            self,
            exc_type,
            exc_value,
            traceback,
        ):
            pass

    monkeypatch.setattr(
        pdf_module.fitz,
        "open",
        lambda *args, **kwargs: FakeDocument(),
    )

    with pytest.raises(
        ValueError,
        match="No readable text could be extracted",
    ):
        pdf_loader.extract_text(file_path)
        
def test_extract_text_propagates_unexpected_error(
    monkeypatch,
    tmp_path,
):
    file_path = tmp_path / "test.pdf"
    file_path.write_bytes(b"fake")

    def raise_unexpected_error(*args, **kwargs):
        raise RuntimeError(
            "Unexpected PDF failure"
        )

    monkeypatch.setattr(
        pdf_module.fitz,
        "open",
        raise_unexpected_error,
    )

    with pytest.raises(
        RuntimeError,
        match="Unexpected PDF failure",
    ):
        pdf_loader.extract_text(
            file_path
        )
        
def test_extract_text_rejects_directory(tmp_path):
    """
    Directories should not be accepted as PDF files.
    """

    directory = tmp_path / "pdf_directory"

    directory.mkdir()

    with pytest.raises(
        ValueError,
        match="PDF path is not a file",
    ):
        pdf_loader.extract_text(
            directory
        )