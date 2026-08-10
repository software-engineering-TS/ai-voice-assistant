from pathlib import Path

from pptx import Presentation

from app.config.logging_config import app_logger


class PptxLoader:
    """
    Extracts text from PowerPoint PPTX presentations.

    Extracts text from:
    - Slide text boxes
    - Titles
    - Content placeholders
    - Other shapes containing text
    """

    # ============================================================
    # Extract Text
    # ============================================================

    def extract_text(
        self,
        file_path: str | Path,
    ) -> str:
        """
        Extract readable text from a PPTX presentation.

        Raises:
            ValueError:
                If the PPTX is invalid, empty, or contains
                no readable text.

            FileNotFoundError:
                If the PPTX file does not exist.
        """

        # --------------------------------------------------------
        # Validate Path
        # --------------------------------------------------------

        if file_path is None:

            raise ValueError(
                "PPTX path cannot be None."
            )
        
        if not str(file_path).strip():
        
            raise ValueError(
                "PPTX path cannot be empty."
            )
        
        path = Path(
            file_path
        )

        if not path.exists():

            raise FileNotFoundError(
                "PPTX file not found."
            )

        if not path.is_file():

            raise ValueError(
                "PPTX path is not a file."
            )

        if path.suffix.lower() != ".pptx":

            raise ValueError(
                "Provided file is not a PPTX file."
            )

        file_name = path.name

        app_logger.info(
            f"Reading PPTX | File={file_name}"
        )

        # --------------------------------------------------------
        # Open Presentation
        # --------------------------------------------------------

        try:

            presentation = Presentation(
                str(path)
            )

        except Exception as exc:

            app_logger.exception(
                f"Failed to open PPTX | "
                f"File={file_name}"
            )

            raise ValueError(
                "The PPTX file is invalid or corrupted."
            ) from exc

        # --------------------------------------------------------
        # Validate Slides
        # --------------------------------------------------------

        slide_count = len(
            presentation.slides
        )

        if slide_count == 0:

            app_logger.warning(
                f"PPTX contains no slides | "
                f"File={file_name}"
            )

            raise ValueError(
                "The PPTX file contains no slides."
            )

        extracted_slides: list[str] = []

        # --------------------------------------------------------
        # Extract Slide Text
        # --------------------------------------------------------

        for slide in presentation.slides:

            slide_text: list[str] = []

            for shape in slide.shapes:

                # Only process shapes containing
                # a text frame.
                if not shape.has_text_frame:
                    continue

                text = shape.text.strip()

                if text:

                    slide_text.append(
                        text
                    )

            if slide_text:

                # Preserve slide boundaries.
                extracted_slides.append(
                    "\n".join(
                        slide_text
                    )
                )

        # --------------------------------------------------------
        # Combine Slides
        # --------------------------------------------------------

        extracted_text = (
            "\n\n".join(
                extracted_slides
            ).strip()
        )

        # --------------------------------------------------------
        # Validate Extracted Text
        # --------------------------------------------------------

        if not extracted_text:

            app_logger.warning(
                f"No readable text extracted from PPTX | "
                f"File={file_name} | "
                f"Slides={slide_count}"
            )

            raise ValueError(
                "No readable text could be extracted "
                "from this PPTX file."
            )

        # --------------------------------------------------------
        # Completed
        # --------------------------------------------------------

        app_logger.info(
            "PPTX text extracted successfully | "
            f"Slides={slide_count} | "
            f"SlidesWithText={len(extracted_slides)} | "
            f"Characters={len(extracted_text)}"
        )

        return extracted_text


# ============================================================
# Shared PPTX Loader
# ============================================================

pptx_loader = PptxLoader()