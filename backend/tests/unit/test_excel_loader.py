from unittest.mock import Mock

import pandas as pd
import pytest

import app.rag.loaders.excel_loader as excel_module
from app.rag.loaders.excel_loader import excel_loader

def test_extract_text_from_single_sheet(tmp_path):
    """
    Verify that data from a single worksheet
    is extracted correctly.
    """

    file_path = tmp_path / "test.xlsx"

    dataframe = pd.DataFrame(
        {
            "Name": ["Alice", "Bob"],
            "Age": [25, 30],
            "Role": ["Engineer", "Developer"],
        }
    )

    dataframe.to_excel(
        file_path,
        index=False,
        sheet_name="Employees",
    )

    result = excel_loader.extract_text(
        str(file_path)
    )

    assert "Sheet: Employees" in result
    assert "Columns: Name, Age, Role" in result
    assert "Name: Alice" in result
    assert "Age: 25" in result
    assert "Role: Engineer" in result
    assert "Name: Bob" in result


def test_extract_text_from_multiple_sheets(tmp_path):
    """
    Verify that all worksheets are processed.
    """

    file_path = tmp_path / "multi_sheet.xlsx"

    employees = pd.DataFrame(
        {
            "Name": ["Alice"],
            "Role": ["Engineer"],
        }
    )

    departments = pd.DataFrame(
        {
            "Department": ["Engineering"],
            "Location": ["Chennai"],
        }
    )

    with pd.ExcelWriter(file_path) as writer:

        employees.to_excel(
            writer,
            index=False,
            sheet_name="Employees",
        )

        departments.to_excel(
            writer,
            index=False,
            sheet_name="Departments",
        )

    result = excel_loader.extract_text(
        str(file_path)
    )

    assert "Sheet: Employees" in result
    assert "Sheet: Departments" in result
    assert "Name: Alice" in result
    assert "Role: Engineer" in result
    assert "Department: Engineering" in result
    assert "Location: Chennai" in result


def test_extract_text_ignores_empty_rows(tmp_path):
    """
    Completely empty rows should not be included
    as document content.
    """

    file_path = tmp_path / "empty_rows.xlsx"

    dataframe = pd.DataFrame(
        {
            "Name": ["Alice", None, "Bob"],
            "Role": ["Engineer", None, "Developer"],
        }
    )

    dataframe.to_excel(
        file_path,
        index=False,
        sheet_name="Employees",
    )

    result = excel_loader.extract_text(
        str(file_path)
    )

    assert "Name: Alice" in result
    assert "Name: Bob" in result
    assert "Role: Engineer" in result
    assert "Role: Developer" in result


def test_extract_text_skips_empty_sheets(tmp_path):
    """
    Empty worksheets should not produce document content.
    """

    file_path = tmp_path / "empty_sheet.xlsx"

    data = pd.DataFrame(
        {
            "Name": ["Alice"],
        }
    )

    empty = pd.DataFrame()

    with pd.ExcelWriter(file_path) as writer:

        data.to_excel(
            writer,
            index=False,
            sheet_name="Data",
        )

        empty.to_excel(
            writer,
            index=False,
            sheet_name="Empty",
        )

    result = excel_loader.extract_text(
        str(file_path)
    )

    assert "Sheet: Data" in result
    assert "Name: Alice" in result

    # Empty sheet should not be represented
    # as readable content.
    assert "Sheet: Empty" not in result


def test_extract_text_rejects_empty_path():
    with pytest.raises(
        ValueError,
        match="Excel path cannot be empty",
    ):
        excel_loader.extract_text("")


def test_extract_text_rejects_missing_file(tmp_path):
    """
    Missing Excel files should raise FileNotFoundError.
    """

    file_path = tmp_path / "missing.xlsx"

    with pytest.raises(FileNotFoundError):

        excel_loader.extract_text(
            str(file_path)
        )


def test_extract_text_rejects_empty_workbook(tmp_path):
    """
    A workbook containing no readable data should
    raise ValueError.
    """

    file_path = tmp_path / "empty.xlsx"

    dataframe = pd.DataFrame()

    with pd.ExcelWriter(file_path) as writer:

        dataframe.to_excel(
            writer,
            index=False,
            sheet_name="Empty",
        )

    with pytest.raises(ValueError):

        excel_loader.extract_text(
            str(file_path)
        )


def test_extract_text_rejects_invalid_excel(tmp_path):
    """
    Invalid Excel content should raise ValueError.
    """

    file_path = tmp_path / "invalid.xlsx"

    file_path.write_bytes(
        b"This is not a valid Excel workbook."
    )

    with pytest.raises(ValueError):

        excel_loader.extract_text(
            str(file_path)
        )


def test_extract_text_cleans_column_names(tmp_path):
    """
    Column names should be stripped of surrounding
    whitespace.
    """

    file_path = tmp_path / "columns.xlsx"

    dataframe = pd.DataFrame(
        {
            " Name ": ["Alice"],
            " Age ": [25],
        }
    )

    dataframe.to_excel(
        file_path,
        index=False,
        sheet_name="Employees",
    )

    result = excel_loader.extract_text(
        str(file_path)
    )

    assert "Columns: Name, Age" in result
    assert "Name: Alice" in result
    assert "Age: 25" in result
    
def test_extract_text_rejects_none_path():
    with pytest.raises(
        ValueError,
        match="Excel path cannot be None",
    ):
        excel_loader.extract_text(None)
        
def test_extract_text_rejects_directory(tmp_path):
    with pytest.raises(
        ValueError,
        match="Excel path is not a file",
    ):
        excel_loader.extract_text(tmp_path)
        
def test_extract_text_rejects_unsupported_file_type(tmp_path):
    file_path = tmp_path / "test.csv"
    file_path.write_text(
        "Name,Age\nAlice,25\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Unsupported Excel file type",
    ):
        excel_loader.extract_text(file_path)
        
def test_extract_text_handles_missing_excel_dependency(
    monkeypatch,
    tmp_path,
):
    file_path = tmp_path / "test.xlsx"

    file_path.write_bytes(b"fake")

    def raise_import_error(*args, **kwargs):
        raise ImportError(
            "openpyxl is not installed"
        )

    monkeypatch.setattr(
        excel_module.pd,
        "read_excel",
        raise_import_error,
    )

    with pytest.raises(
        RuntimeError,
        match="Required Excel reader dependency is missing",
    ):
        excel_module.excel_loader.extract_text(
            file_path
        )
        
def test_extract_text_handles_os_error(
    monkeypatch,
    tmp_path,
):
    file_path = tmp_path / "test.xlsx"
    file_path.write_bytes(b"fake")

    def raise_os_error(*args, **kwargs):
        raise OSError("permission denied")

    monkeypatch.setattr(
        excel_module.pd,
        "read_excel",
        raise_os_error,
    )

    with pytest.raises(
        ValueError,
        match="Unable to read the Excel file",
    ):
        excel_loader.extract_text(file_path)
        
def test_extract_text_propagates_unexpected_error(
    monkeypatch,
    tmp_path,
):
    file_path = tmp_path / "test.xlsx"
    file_path.write_bytes(b"fake")

    def raise_unexpected_error(*args, **kwargs):
        raise RuntimeError("unexpected failure")

    monkeypatch.setattr(
        excel_module.pd,
        "read_excel",
        raise_unexpected_error,
    )

    with pytest.raises(
        RuntimeError,
        match="unexpected failure",
    ):
        excel_loader.extract_text(file_path)
        
def test_extract_text_rejects_workbook_with_no_worksheets(
    monkeypatch,
    tmp_path,
):
    file_path = tmp_path / "test.xlsx"
    file_path.write_bytes(b"fake")

    monkeypatch.setattr(
        excel_module.pd,
        "read_excel",
        lambda *args, **kwargs: {},
    )

    with pytest.raises(
        ValueError,
        match="The Excel workbook contains no worksheets",
    ):
        excel_loader.extract_text(file_path)
        
def test_extract_text_skips_none_worksheet(
    monkeypatch,
    tmp_path,
):
    file_path = tmp_path / "test.xlsx"
    file_path.write_bytes(b"fake")

    dataframe = pd.DataFrame(
        {
            "Name": ["Alice"],
        }
    )

    monkeypatch.setattr(
        excel_module.pd,
        "read_excel",
        lambda *args, **kwargs: {
            "EmptySheet": None,
            "Data": dataframe,
        },
    )

    result = excel_loader.extract_text(file_path)

    assert "Sheet: Data" in result
    assert "Name: Alice" in result
    assert "Sheet: EmptySheet" not in result
    
def test_extract_text_rejects_no_readable_content(
    monkeypatch,
    tmp_path,
):
    file_path = tmp_path / "test.xlsx"
    file_path.write_bytes(b"fake")

    dataframe = pd.DataFrame(
        {
            "Name": ["", ""],
            "Role": ["", ""],
        }
    )

    monkeypatch.setattr(
        excel_module.pd,
        "read_excel",
        lambda *args, **kwargs: {
            "Employees": dataframe,
        },
    )

    with pytest.raises(
        ValueError,
        match="No readable content could be extracted",
    ):
        excel_loader.extract_text(file_path)
        
def test_extract_text_skips_nan_values(
    monkeypatch,
    tmp_path,
):
    file_path = tmp_path / "test.xlsx"
    file_path.write_bytes(b"fake")

    dataframe = pd.DataFrame(
        {
            "Name": ["Alice"],
            "Age": [None],
            "Role": ["Engineer"],
        }
    )

    monkeypatch.setattr(
        excel_module.pd,
        "read_excel",
        lambda *args, **kwargs: {
            "Employees": dataframe,
        },
    )

    result = excel_loader.extract_text(file_path)

    assert "Name: Alice" in result
    assert "Role: Engineer" in result
    assert "Age:" not in result
    

def test_extract_text_handles_missing_excel_dependency(
    monkeypatch,
    tmp_path,
):
    file_path = tmp_path / "test.xlsx"

    file_path.write_bytes(b"fake")

    def raise_import_error(*args, **kwargs):
        raise ImportError(
            "openpyxl is not installed"
        )

    monkeypatch.setattr(
        excel_module.pd,
        "read_excel",
        raise_import_error,
    )

    with pytest.raises(
        RuntimeError,
        match="Required Excel reader dependency is missing",
    ):
        excel_loader.extract_text(
            file_path
        )