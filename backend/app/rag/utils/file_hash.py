import hashlib
from pathlib import Path

from app.config.logging_config import app_logger


class FileHash:
    """
    Utility for generating cryptographic hashes for files.

    SHA-256 is used for document duplicate detection.
    """

    HASH_ALGORITHM = "sha256"
    BUFFER_SIZE = 1024 * 1024  # 1 MB

    # ============================================================
    # SHA-256
    # ============================================================

    @classmethod
    def sha256(
        cls,
        file_path: str | Path,
    ) -> str:
        """
        Calculate the SHA-256 hash of a file.

        Parameters:
            file_path:
                Path to the file.

        Returns:
            SHA-256 hexadecimal digest.
        """

        if file_path is None:

            raise ValueError(
                "File path cannot be None."
            )

        path = Path(file_path)

        if not path.exists():

            raise FileNotFoundError(
                f"File not found: {path}"
            )

        if not path.is_file():

            raise ValueError(
                f"Path is not a file: {path}"
            )

        app_logger.debug(
            f"Calculating SHA-256 hash | "
            f"File={path.name}"
        )

        sha = hashlib.sha256()

        try:

            with path.open(
                "rb"
            ) as file:

                while True:

                    chunk = file.read(
                        cls.BUFFER_SIZE
                    )

                    if not chunk:
                        break

                    sha.update(chunk)

        except OSError:

            app_logger.exception(
                f"Failed to calculate file hash | "
                f"File={path.name}"
            )

            raise

        file_hash = sha.hexdigest()

        app_logger.debug(
            f"SHA-256 hash calculated | "
            f"File={path.name}"
        )

        return file_hash


# ============================================================
# Shared File Hash Utility
# ============================================================

file_hash = FileHash()