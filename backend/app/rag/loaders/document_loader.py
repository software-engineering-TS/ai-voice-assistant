from pathlib import Path

from app.config.logging_config import app_logger

from app.rag.loaders.csv_loader import csv_loader
from app.rag.loaders.docx_loader import docx_loader
from app.rag.loaders.excel_loader import excel_loader
from app.rag.loaders.pdf_loader import pdf_loader
from app.rag.loaders.pptx_loader import pptx_loader
from app.rag.loaders.txt_loader import txt_loader


# ============================================================
# Supported Document Loaders
# ============================================================

LOADERS = {
    ".pdf": pdf_loader,
    ".docx": docx_loader,
    ".pptx": pptx_loader,
    ".txt": txt_loader,
    ".csv": csv_loader,
    ".xlsx": excel_loader,
    ".xls": excel_loader,
}


class DocumentLoader:
    """
    Selects the appropriate document loader based on the
    file extension and extracts text from the file.
    """

    # ========================================================
    # Extract Text
    # ========================================================

    def extract_text(
        self,
        file_path: str | Path,
    ) -> str:
        """
        Extract text from a supported document.

        Raises:
            ValueError:
                If the path or extension is invalid.

            FileNotFoundError:
                If the document does not exist.

            RuntimeError:
                If the underlying loader fails.
        """

        # ----------------------------------------------------
        # Validate Path
        # ----------------------------------------------------

        if file_path is None:

            raise ValueError(
                "File path cannot be None."
            )

        path = Path(
            file_path
        )

        if not str(path).strip():

            raise ValueError(
                "File path cannot be empty."
            )

        if not path.exists():

            raise FileNotFoundError(
                "Document not found."
            )

        if not path.is_file():

            raise ValueError(
                "Document path is not a file."
            )

        # ----------------------------------------------------
        # Determine Extension
        # ----------------------------------------------------

        extension = (
            path.suffix
            .lower()
        )

        if not extension:

            raise ValueError(
                "Document does not have a file extension."
            )

        app_logger.info(
            "Document loader selected | "
            f"Extension={extension}"
        )

        # ----------------------------------------------------
        # Find Loader
        # ----------------------------------------------------

        loader = LOADERS.get(
            extension
        )

        if loader is None:

            app_logger.warning(
                "Unsupported document type | "
                f"Extension={extension}"
            )

            raise ValueError(
                f"Unsupported file type: {extension}"
            )

        # ----------------------------------------------------
        # Extract Text
        # ----------------------------------------------------

        try:

            text = loader.extract_text(
                str(path)
            )

        except Exception:

            app_logger.exception(
                "Document extraction failed | "
                f"Extension={extension}"
            )

            raise

        # ----------------------------------------------------
        # Validate Extracted Text
        # ----------------------------------------------------

        if not isinstance(
            text,
            str,
        ):

            app_logger.error(
                "Document loader returned "
                "a non-string result."
            )

            raise ValueError(
                "Document loader returned invalid text."
            )

        text = text.strip()

        if not text:

            app_logger.warning(
                "No readable text extracted | "
                f"Extension={extension}"
            )

            raise ValueError(
                "No readable text could be extracted "
                "from the document."
            )

        # ----------------------------------------------------
        # Completed
        # ----------------------------------------------------

        app_logger.info(
            "Document extraction completed | "
            f"Characters={len(text)} | "
            f"Extension={extension}"
        )

        return text


# ============================================================
# Shared Document Loader
# ============================================================

document_loader = DocumentLoader()