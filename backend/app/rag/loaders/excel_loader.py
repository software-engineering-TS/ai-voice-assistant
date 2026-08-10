from pathlib import Path

import pandas as pd

from app.config.logging_config import app_logger


class ExcelLoader:
    """
    Extracts structured data from Excel workbooks.

    Supports:
    - .xlsx
    - .xls

    Each worksheet is converted into structured text suitable
    for downstream chunking and semantic retrieval.
    """

    SUPPORTED_EXTENSIONS = {
        ".xlsx",
        ".xls",
    }

    # ============================================================
    # Extract Text
    # ============================================================

    def extract_text(
        self,
        file_path: str | Path,
    ) -> str:
        """
        Read all worksheets from an Excel workbook and convert
        their contents into structured text.
        """

        # --------------------------------------------------------
        # Validate Path
        # --------------------------------------------------------

        if file_path is None:
            raise ValueError(
                "Excel path cannot be None."
            )

        if not str(file_path).strip():
            raise ValueError(
                "Excel path cannot be empty."
            )
        
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                "Excel file not found."
            )

        if not path.is_file():
            raise ValueError(
                "Excel path is not a file."
            )

        extension = path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                "Unsupported Excel file type."
            )

        file_name = path.name

        app_logger.info(
            f"Reading Excel | File={file_name}"
        )

        # --------------------------------------------------------
        # Load Workbook
        # --------------------------------------------------------

        try:
            sheets = pd.read_excel(
                path,
                sheet_name=None,
                dtype=str,
            )

        except ImportError as exc:
            app_logger.exception(
                "Required Excel reader dependency "
                "is not installed."
            )

            raise RuntimeError(
                "Required Excel reader dependency "
                "is missing."
            ) from exc

        except ValueError as exc:
            app_logger.exception(
                f"Invalid Excel workbook | "
                f"File={file_name}"
            )

            raise ValueError(
                "The Excel file could not be read."
            ) from exc

        except OSError as exc:
            app_logger.exception(
                f"Failed to read Excel file | "
                f"File={file_name}"
            )

            raise ValueError(
                "Unable to read the Excel file."
            ) from exc

        except Exception:
            app_logger.exception(
                f"Excel extraction failed | "
                f"File={file_name}"
            )

            raise

        # --------------------------------------------------------
        # Validate Workbook
        # --------------------------------------------------------

        if not sheets:
            app_logger.warning(
                f"Excel workbook contains no worksheets | "
                f"File={file_name}"
            )

            raise ValueError(
                "The Excel workbook contains no worksheets."
            )

        # --------------------------------------------------------
        # Process Worksheets
        # --------------------------------------------------------

        extracted_parts: list[str] = []

        total_rows = 0
        sheets_with_data = 0

        for sheet_name, dataframe in sheets.items():

            sheet_name = str(
                sheet_name
            ).strip()

            # ----------------------------------------------------
            # Skip None Worksheet
            # ----------------------------------------------------

            if dataframe is None:
                continue

            # ----------------------------------------------------
            # Remove Completely Empty Rows
            # ----------------------------------------------------

            dataframe = dataframe.dropna(
                how="all"
            )

            if dataframe.empty:
                continue

            # ----------------------------------------------------
            # Clean Column Names
            # ----------------------------------------------------

            dataframe.columns = [
                str(column).strip()
                for column in dataframe.columns
            ]

            columns = list(
                dataframe.columns
            )

            # ----------------------------------------------------
            # Extract Readable Rows
            # ----------------------------------------------------

            sheet_rows: list[str] = []

            for row_number, row in enumerate(
                dataframe.itertuples(
                    index=False,
                    name=None,
                ),
                start=1,
            ):

                row_values: list[str] = []

                for column, value in zip(
                    columns,
                    row,
                ):

                    # Skip NaN values
                    if pd.isna(value):
                        continue

                    value = str(
                        value
                    ).strip()

                    # Skip empty values
                    if not value:
                        continue

                    row_values.append(
                        f"{column}: {value}"
                    )

                # Only add rows containing readable values
                if row_values:

                    sheet_rows.append(
                        f"Row {row_number}: "
                        + " | ".join(
                            row_values
                        )
                    )

            # ----------------------------------------------------
            # Skip Worksheet With No Readable Content
            # ----------------------------------------------------

            if not sheet_rows:
                continue

            # ----------------------------------------------------
            # Worksheet Contains Readable Content
            # ----------------------------------------------------

            sheets_with_data += 1
            total_rows += len(
                dataframe
            )

            # ----------------------------------------------------
            # Sheet Header
            # ----------------------------------------------------

            extracted_parts.append(
                f"Sheet: {sheet_name}"
            )

            extracted_parts.append(
                "Columns: "
                + ", ".join(columns)
            )

            # ----------------------------------------------------
            # Add Extracted Rows
            # ----------------------------------------------------

            extracted_parts.extend(
                sheet_rows
            )

            # ----------------------------------------------------
            # Strong Boundary Between Worksheets
            # ----------------------------------------------------

            extracted_parts.append("")

        # --------------------------------------------------------
        # Build Final Text
        # --------------------------------------------------------

        extracted_text = "\n".join(
            extracted_parts
        ).strip()

        # --------------------------------------------------------
        # Validate Extracted Content
        # --------------------------------------------------------

        if not extracted_text:

            app_logger.warning(
                f"No readable content extracted from Excel | "
                f"File={file_name}"
            )

            raise ValueError(
                "No readable content could be extracted "
                "from the Excel workbook."
            )

        # --------------------------------------------------------
        # Completed
        # --------------------------------------------------------

        app_logger.info(
            "Excel extracted successfully | "
            f"Sheets={len(sheets)} | "
            f"SheetsWithData={sheets_with_data} | "
            f"Rows={total_rows} | "
            f"Characters={len(extracted_text)}"
        )

        return extracted_text


# ============================================================
# Shared Excel Loader
# ============================================================

excel_loader = ExcelLoader()