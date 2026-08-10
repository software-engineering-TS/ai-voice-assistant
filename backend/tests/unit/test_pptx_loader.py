import pytest
from pptx import Presentation
import app.rag.loaders.pptx_loader as pptx_module
from app.rag.loaders.pptx_loader import pptx_loader


def test_extract_text_from_slides(tmp_path):
    """
    Verify that text from multiple slides is extracted.
    """

    file_path = tmp_path / "test.pptx"

    presentation = Presentation()

    slide = presentation.slides.add_slide(
        presentation.slide_layouts[1]
    )

    slide.shapes.title.text = "Introduction"

    slide.placeholders[1].text = (
        "This is the first slide."
    )

    slide = presentation.slides.add_slide(
        presentation.slide_layouts[1]
    )

    slide.shapes.title.text = "Architecture"

    slide.placeholders[1].text = (
        "This is the second slide."
    )

    presentation.save(file_path)

    result = pptx_loader.extract_text(
        str(file_path)
    )

    assert "Introduction" in result
    assert "This is the first slide." in result
    assert "Architecture" in result
    assert "This is the second slide." in result


def test_extract_text_from_text_box(tmp_path):
    """
    Verify that text boxes are extracted.
    """

    file_path = tmp_path / "textbox.pptx"

    presentation = Presentation()

    slide = presentation.slides.add_slide(
        presentation.slide_layouts[6]
    )

    textbox = slide.shapes.add_textbox(
        100,
        100,
        500,
        200,
    )

    textbox.text = "Text inside a text box."

    presentation.save(file_path)

    result = pptx_loader.extract_text(
        str(file_path)
    )

    assert "Text inside a text box." in result


def test_extract_text_preserves_slide_boundaries(
    tmp_path,
):
    """
    Text from separate slides should remain separated.
    """

    file_path = tmp_path / "slides.pptx"

    presentation = Presentation()

    slide1 = presentation.slides.add_slide(
        presentation.slide_layouts[6]
    )

    slide1.shapes.add_textbox(
        100,
        100,
        500,
        100,
    ).text = "Slide One"

    slide2 = presentation.slides.add_slide(
        presentation.slide_layouts[6]
    )

    slide2.shapes.add_textbox(
        100,
        100,
        500,
        100,
    ).text = "Slide Two"

    presentation.save(file_path)

    result = pptx_loader.extract_text(
        str(file_path)
    )

    assert "Slide One" in result
    assert "Slide Two" in result
    assert "Slide One\n\nSlide Two" in result


def test_extract_text_ignores_shapes_without_text(
    tmp_path,
):
    """
    Shapes without text should not cause errors.
    """

    file_path = tmp_path / "shapes.pptx"

    presentation = Presentation()

    slide = presentation.slides.add_slide(
        presentation.slide_layouts[6]
    )

    slide.shapes.add_shape(
        1,
        100,
        100,
        200,
        200,
    )

    textbox = slide.shapes.add_textbox(
        100,
        350,
        500,
        100,
    )

    textbox.text = "Valid text."

    presentation.save(file_path)

    result = pptx_loader.extract_text(
        str(file_path)
    )

    assert "Valid text." in result


def test_extract_text_rejects_empty_path():
    """
    Empty paths should raise ValueError.
    """

    with pytest.raises(ValueError):

        pptx_loader.extract_text("")


def test_extract_text_rejects_missing_file(tmp_path):
    """
    Missing PPTX files should raise FileNotFoundError.
    """

    file_path = tmp_path / "missing.pptx"

    with pytest.raises(FileNotFoundError):

        pptx_loader.extract_text(
            str(file_path)
        )


def test_extract_text_rejects_empty_presentation(
    tmp_path,
):
    """
    A presentation containing no readable text
    should raise ValueError.
    """

    file_path = tmp_path / "empty.pptx"

    presentation = Presentation()

    presentation.slides.add_slide(
        presentation.slide_layouts[6]
    )

    presentation.save(file_path)

    with pytest.raises(
        ValueError,
        match="No readable text could be extracted",
    ):
        pptx_loader.extract_text(file_path)


def test_extract_text_rejects_invalid_pptx(
    tmp_path,
):
    """
    Invalid PPTX content should raise ValueError.
    """

    file_path = tmp_path / "invalid.pptx"

    file_path.write_bytes(
        b"This is not a valid PPTX file."
    )

    with pytest.raises(ValueError):

        pptx_loader.extract_text(
            str(file_path)
        )
        

def test_extract_text_rejects_none_path():
    """
    None PPTX paths should be rejected.
    """

    with pytest.raises(
        ValueError,
        match="PPTX path cannot be None",
    ):
        pptx_loader.extract_text(None)
        
def test_extract_text_rejects_empty_path():
    """
    Empty PPTX paths should be rejected.
    """

    with pytest.raises(
        ValueError,
        match="PPTX path cannot be empty",
    ):
        pptx_loader.extract_text("")
        
def test_extract_text_rejects_non_pptx_file(tmp_path):
    """
    Non-PPTX files should be rejected.
    """

    file_path = tmp_path / "test.txt"

    file_path.write_text(
        "This is not a PPTX file.",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Provided file is not a PPTX file",
    ):
        pptx_loader.extract_text(file_path)
        
def test_extract_text_skips_shape_without_text_frame(
    monkeypatch,
    tmp_path,
):
    """
    Shapes without a text frame should be ignored.
    """

    file_path = tmp_path / "shapes.pptx"
    file_path.write_bytes(b"fake")

    class NoTextShape:
        has_text_frame = False

    class TextShape:
        has_text_frame = True
        text = "Valid text."

    class FakeSlide:
        shapes = [
            NoTextShape(),
            TextShape(),
        ]

    class FakePresentation:
        slides = [FakeSlide()]

    monkeypatch.setattr(
        pptx_module,
        "Presentation",
        lambda *args, **kwargs: FakePresentation(),
    )

    result = pptx_loader.extract_text(
        file_path
    )

    assert result == "Valid text."
    
def test_extract_text_rejects_pptx_with_no_slides(
    tmp_path,
):
    """
    A PPTX containing no slides should be rejected.
    """

    file_path = tmp_path / "no_slides.pptx"

    presentation = Presentation()

    # Do NOT add any slides.
    presentation.save(file_path)

    with pytest.raises(
        ValueError,
        match="The PPTX file contains no slides",
    ):
        pptx_loader.extract_text(file_path)
        
def test_extract_text_rejects_directory(tmp_path):
    """
    A directory path should be rejected.
    """

    directory = tmp_path / "pptx_directory"
    directory.mkdir()

    with pytest.raises(
        ValueError,
        match="PPTX path is not a file",
    ):
        pptx_loader.extract_text(directory)