from pathlib import Path

from app.config.logging_config import app_logger


class TxtLoader:
    """
    Extracts text from plain-text files.
    """

    # ============================================================
    # Extract Text
    # ============================================================

    def extract_text(
        self,
        file_path: str | Path,
    ) -> str:
        """
        Read and return text from a TXT file.

        The file is expected to contain UTF-8 encoded text.
        """

        # --------------------------------------------------------
        # Validate Path
        # --------------------------------------------------------

        if file_path is None:

            raise ValueError(
                "TXT path cannot be None."
            )

        path = Path(
            file_path
        )

        if not str(path).strip():

            raise ValueError(
                "TXT path cannot be empty."
            )

        if not path.exists():

            raise FileNotFoundError(
                "TXT file not found."
            )

        if not path.is_file():

            raise ValueError(
                "TXT path is not a file."
            )

        if path.suffix.lower() != ".txt":

            raise ValueError(
                "Provided file is not a TXT file."
            )

        file_name = path.name

        app_logger.info(
            f"Reading TXT | File={file_name}"
        )

        # --------------------------------------------------------
        # Read File
        # --------------------------------------------------------

        try:

            text = path.read_text(
                encoding="utf-8"
            )

        except UnicodeDecodeError as exc:

            app_logger.exception(
                f"TXT encoding error | "
                f"File={file_name}"
            )

            raise ValueError(
                "The TXT file is not valid UTF-8 text."
            ) from exc

        except OSError as exc:

            app_logger.exception(
                f"Failed to read TXT | "
                f"File={file_name}"
            )

            raise ValueError(
                "Unable to read the TXT file."
            ) from exc

        # --------------------------------------------------------
        # Clean Text
        # --------------------------------------------------------

        text = text.strip()

        if not text:

            app_logger.warning(
                f"Empty TXT file | "
                f"File={file_name}"
            )

            raise ValueError(
                "The TXT file contains no readable text."
            )

        # --------------------------------------------------------
        # Completed
        # --------------------------------------------------------

        app_logger.info(
            "TXT extracted successfully | "
            f"Characters={len(text)}"
        )

        return text


# ============================================================
# Shared TXT Loader
# ============================================================

txt_loader = TxtLoader()