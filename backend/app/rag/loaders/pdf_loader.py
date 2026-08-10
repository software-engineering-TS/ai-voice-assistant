from pathlib import Path

import fitz

from app.config.logging_config import app_logger


class PDFLoader:
    """
    Extracts text from PDF documents using PyMuPDF.
    """

    # ============================================================
    # Extract Text
    # ============================================================

    def extract_text(
        self,
        pdf_path: str | Path,
    ) -> str:
        """
        Extract text from all readable pages of a PDF.

        Raises:
            ValueError:
                If the PDF is invalid, empty, or contains no
                readable text.

            FileNotFoundError:
                If the PDF does not exist.
        """

        # --------------------------------------------------------
        # Validate Path
        # --------------------------------------------------------
        
        if pdf_path is None:
            raise ValueError(
                "PDF path cannot be None."
            )
        
        if not str(pdf_path).strip():
            raise ValueError(
                "PDF path cannot be empty."
            )
        
        path = Path(
            pdf_path
        )
        
        if not path.exists():
            raise FileNotFoundError(
                "PDF file not found."
            )
        
        if not path.is_file():
            raise ValueError(
                "PDF path is not a file."
            )
        
        if path.suffix.lower() != ".pdf":
            raise ValueError(
                "Provided file is not a PDF."
            )

        file_name = path.name

        app_logger.info(
            f"Reading PDF | File={file_name}"
        )

        # --------------------------------------------------------
        # Open and Extract
        # --------------------------------------------------------

        page_count = 0
        page_text: list[str] = []

        try:

            with fitz.open(
                str(path)
            ) as document:

                page_count = len(
                    document
                )

                app_logger.info(
                    f"PDF opened successfully | "
                    f"Pages={page_count}"
                )

                if page_count == 0:

                    raise ValueError(
                        "PDF contains no pages."
                    )

                for page_number, page in enumerate(
                    document,
                    start=1,
                ):

                    try:

                        text = page.get_text(
                            "text"
                        )

                    except Exception:

                        app_logger.exception(
                            "Failed to extract PDF page | "
                            f"Page={page_number} | "
                            f"File={file_name}"
                        )

                        continue

                    if not text:
                        continue

                    text = text.strip()

                    if not text:
                        continue

                    page_text.append(
                        text
                    )

        except fitz.FileDataError as exc:

            app_logger.exception(
                f"Invalid or corrupted PDF | "
                f"File={file_name}"
            )

            raise ValueError(
                "The PDF file is invalid or corrupted."
            ) from exc

        except ValueError:

            raise

        except Exception:

            app_logger.exception(
                f"PDF extraction failed | "
                f"File={file_name}"
            )

            raise

        # --------------------------------------------------------
        # Validate Extracted Text
        # --------------------------------------------------------

        extracted_text = (
            "\n\n".join(
                page_text
            ).strip()
        )

        if not extracted_text:

            app_logger.warning(
                f"No readable text extracted from PDF | "
                f"File={file_name} | "
                f"Pages={page_count}"
            )

            raise ValueError(
                "No readable text could be extracted "
                "from this PDF. The document may be "
                "image-based or scanned."
            )

        # --------------------------------------------------------
        # Completed
        # --------------------------------------------------------

        app_logger.info(
            f"PDF text extracted successfully | "
            f"Pages={page_count} | "
            f"Characters={len(extracted_text)}"
        )

        return extracted_text


# ============================================================
# Shared PDF Loader
# ============================================================

pdf_loader = PDFLoader()