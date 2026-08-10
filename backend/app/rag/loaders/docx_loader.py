from pathlib import Path

from docx import Document

from app.config.logging_config import app_logger


class DocxLoader:
    """
    Extracts text from DOCX documents.

    Extracts:
    - Paragraphs
    - Table cells
    """

    # ============================================================
    # Extract Text
    # ============================================================

    def extract_text(
        self,
        file_path: str | Path,
    ) -> str:
        """
        Extract readable text from a DOCX file.

        Raises:
            ValueError:
                If the file is invalid, unsupported, or empty.

            FileNotFoundError:
                If the DOCX file does not exist.
        """

        # --------------------------------------------------------
        # Validate Path
        # --------------------------------------------------------

        if file_path is None:

            raise ValueError(
                "DOCX path cannot be None."
            )

        if not str(file_path).strip():

            raise ValueError(
                "DOCX path cannot be empty."
            )
        
        path = Path(
            file_path
        )

        if not path.exists():

            raise FileNotFoundError(
                "DOCX file not found."
            )

        if not path.is_file():

            raise ValueError(
                "DOCX path is not a file."
            )

        if path.suffix.lower() != ".docx":

            raise ValueError(
                "Provided file is not a DOCX file."
            )

        file_name = path.name

        app_logger.info(
            f"Reading DOCX | File={file_name}"
        )

        # --------------------------------------------------------
        # Open DOCX
        # --------------------------------------------------------

        try:

            document = Document(
                str(path)
            )

        except Exception as exc:

            app_logger.exception(
                f"Failed to open DOCX | "
                f"File={file_name}"
            )

            raise ValueError(
                "The DOCX file is invalid or corrupted."
            ) from exc

        extracted_parts: list[str] = []

        # --------------------------------------------------------
        # Extract Paragraphs
        # --------------------------------------------------------

        for paragraph in (
            document.paragraphs
        ):

            text = paragraph.text.strip()

            if text:

                extracted_parts.append(
                    text
                )

        # --------------------------------------------------------
        # Extract Tables
        # --------------------------------------------------------

        for table_index, table in enumerate(
            document.tables,
            start=1,
        ):

            app_logger.debug(
                f"Reading DOCX table | "
                f"Table={table_index}"
            )

            for row in table.rows:

                row_cells: list[str] = []

                for cell in row.cells:

                    cell_text = (
                        cell.text.strip()
                    )

                    if cell_text:

                        row_cells.append(
                            cell_text
                        )

                if row_cells:

                    extracted_parts.append(
                        " | ".join(
                            row_cells
                        )
                    )

        # --------------------------------------------------------
        # Build Final Text
        # --------------------------------------------------------

        extracted_text = (
            "\n".join(
                extracted_parts
            ).strip()
        )

        # --------------------------------------------------------
        # Validate Result
        # --------------------------------------------------------

        if not extracted_text:

            app_logger.warning(
                "No readable text extracted from DOCX | "
                f"File={file_name}"
            )

            raise ValueError(
                "No readable text could be extracted "
                "from this DOCX file."
            )

        # --------------------------------------------------------
        # Completed
        # --------------------------------------------------------

        app_logger.info(
            "DOCX text extracted successfully | "
            f"Characters={len(extracted_text)} | "
            f"Paragraphs={len(document.paragraphs)} | "
            f"Tables={len(document.tables)}"
        )

        return extracted_text


# ============================================================
# Shared DOCX Loader
# ============================================================

docx_loader = DocxLoader()