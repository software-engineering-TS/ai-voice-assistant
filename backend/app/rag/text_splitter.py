from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from app.config.logging_config import app_logger
from app.config.settings import settings


class TextSplitterService:
    """
    Splits extracted document text into overlapping chunks
    suitable for embedding and vector retrieval.
    """

    def __init__(self) -> None:

        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = (
            settings.CHUNK_OVERLAP
        )

        # --------------------------------------------------------
        # Validate Configuration
        # --------------------------------------------------------

        if self.chunk_size <= 0:

            raise ValueError(
                "CHUNK_SIZE must be greater than zero."
            )

        if self.chunk_overlap < 0:

            raise ValueError(
                "CHUNK_OVERLAP cannot be negative."
            )

        if self.chunk_overlap >= self.chunk_size:

            raise ValueError(
                "CHUNK_OVERLAP must be smaller "
                "than CHUNK_SIZE."
            )

        # --------------------------------------------------------
        # Initialize Splitter
        # --------------------------------------------------------

        self.splitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                separators=[
                    "\n\n",
                    "\n",
                    ". ",
                    " ",
                    "",
                ],
            )
        )

        app_logger.info(
            "Text splitter initialized | "
            f"ChunkSize={self.chunk_size} | "
            f"ChunkOverlap={self.chunk_overlap}"
        )

    # ============================================================
    # Split Text
    # ============================================================

    def split(
        self,
        text: str,
    ) -> list[str]:
        """
        Split document text into chunks.

        Returns:
            A list of non-empty text chunks.
        """

        # --------------------------------------------------------
        # Validate Input
        # --------------------------------------------------------

        if not isinstance(
            text,
            str,
        ):

            raise TypeError(
                "Document text must be a string."
            )

        text = text.strip()

        if not text:

            app_logger.warning(
                "Empty document text received."
            )

            return []

        app_logger.info(
            "Splitting document | "
            f"Characters={len(text)}"
        )

        # --------------------------------------------------------
        # Split Document
        # --------------------------------------------------------

        try:

            chunks = (
                self.splitter.split_text(
                    text
                )
            )

        except Exception:

            app_logger.exception(
                "Document text splitting failed."
            )

            raise

        # --------------------------------------------------------
        # Clean Chunks
        # --------------------------------------------------------

        cleaned_chunks: list[str] = []

        for chunk in chunks:

            if not chunk:
                continue

            chunk = chunk.strip()

            if not chunk:
                continue

            cleaned_chunks.append(
                chunk
            )

        # --------------------------------------------------------
        # Validate Result
        # --------------------------------------------------------

        if not cleaned_chunks:

            app_logger.warning(
                "Text splitter produced no chunks."
            )

            return []

        # --------------------------------------------------------
        # Statistics
        # --------------------------------------------------------

        total_characters = sum(
            len(chunk)
            for chunk in cleaned_chunks
        )

        average_size = (
            total_characters
            / len(cleaned_chunks)
        )

        largest_chunk = max(
            len(chunk)
            for chunk in cleaned_chunks
        )

        smallest_chunk = min(
            len(chunk)
            for chunk in cleaned_chunks
        )

        app_logger.info(
            "Document splitting completed | "
            f"Chunks={len(cleaned_chunks)} | "
            f"AverageSize={average_size:.0f} | "
            f"MinSize={smallest_chunk} | "
            f"MaxSize={largest_chunk}"
        )

        return cleaned_chunks


# ============================================================
# Shared Service Instance
# ============================================================

text_splitter_service = TextSplitterService()