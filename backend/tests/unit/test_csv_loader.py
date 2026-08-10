import pandas as pd
import pytest
from pathlib import Path

import app.rag.loaders.csv_loader as csv_module
from app.rag.loaders.csv_loader import csv_loader


def test_extract_text_successfully(tmp_path):
    """
    Verify that CSV data is converted into structured text.
    """

    file_path = tmp_path / "test.csv"

    dataframe = pd.DataFrame(
        {
            "Name": ["Alice", "Bob"],
            "Age": ["25", "30"],
            "Role": ["Engineer", "Developer"],
        }
    )

    dataframe.to_csv(
        file_path,
        index=False,
    )

    result = csv_loader.extract_text(
        str(file_path)
    )

    assert "Columns: Name, Age, Role" in result
    assert "Row 1:" in result
    assert "Name: Alice" in result
    assert "Age: 25" in result
    assert "Role: Engineer" in result
    assert "Row 2:" in result
    assert "Name: Bob" in result


def test_extract_text_handles_empty_values(tmp_path):
    """
    Empty CSV values should not produce meaningless
    field entries.
    """

    file_path = tmp_path / "test.csv"

    dataframe = pd.DataFrame(
        {
            "Name": ["Alice"],
            "Age": [""],
            "Role": ["Engineer"],
        }
    )

    dataframe.to_csv(
        file_path,
        index=False,
    )

    result = csv_loader.extract_text(
        str(file_path)
    )

    assert "Name: Alice" in result
    assert "Role: Engineer" in result
    assert "Age:" not in result


def test_extract_text_rejects_empty_path(monkeypatch):
    class FakePath:
        def __init__(self, value):
            self.value = value

        def __str__(self):
            return ""

    monkeypatch.setattr(
        csv_module,
        "Path",
        FakePath,
    )

    with pytest.raises(
        ValueError,
        match="CSV path cannot be empty.",
    ):
        csv_loader.extract_text("")


def test_extract_text_rejects_missing_file(tmp_path):
    """
    Missing CSV files should raise FileNotFoundError.
    """

    file_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):

        csv_loader.extract_text(
            str(file_path)
        )


def test_extract_text_rejects_empty_csv(tmp_path):
    """
    A completely empty CSV should raise ValueError.
    """

    file_path = tmp_path / "empty.csv"

    file_path.write_text(
        "",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):

        csv_loader.extract_text(
            str(file_path)
        )


def test_extract_text_rejects_csv_with_no_rows(tmp_path):
    """
    A CSV containing only column headers and no rows
    should be rejected.
    """

    file_path = tmp_path / "headers.csv"

    file_path.write_text(
        "Name,Age,Role\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):

        csv_loader.extract_text(
            str(file_path)
        )


def test_extract_text_cleans_column_names(tmp_path):
    """
    Column names should be stripped of surrounding
    whitespace.
    """

    file_path = tmp_path / "test.csv"

    file_path.write_text(
        " Name , Age \nAlice,25\n",
        encoding="utf-8",
    )

    result = csv_loader.extract_text(
        str(file_path)
    )

    assert "Columns: Name, Age" in result
    assert "Name: Alice" in result
    assert "Age: 25" in result
    
def test_extract_text_rejects_none_path():
    with pytest.raises(
        ValueError,
        match="CSV path cannot be None.",
    ):
        csv_loader.extract_text(None)
        
def test_extract_text_rejects_directory(tmp_path):
    directory = tmp_path / "data"
    directory.mkdir()

    with pytest.raises(
        ValueError,
        match="CSV path is not a file.",
    ):
        csv_loader.extract_text(directory)
        
def test_extract_text_rejects_non_csv_file(tmp_path):
    file_path = tmp_path / "test.txt"

    file_path.write_text(
        "hello",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Provided file is not a CSV file.",
    ):
        csv_loader.extract_text(file_path)

def test_extract_text_handles_unicode_decode_error(
    tmp_path,
    monkeypatch,
):
    file_path = tmp_path / "test.csv"

    file_path.write_text(
        "Name\nAlice\n",
        encoding="utf-8",
    )

    def mock_read_csv(*args, **kwargs):
        raise UnicodeDecodeError(
            "utf-8",
            b"\xff",
            0,
            1,
            "invalid start byte",
        )

    monkeypatch.setattr(
        csv_module.pd,
        "read_csv",
        mock_read_csv,
    )

    with pytest.raises(
        ValueError,
        match="encoding could not be read as UTF-8",
    ):
        csv_loader.extract_text(file_path)
        
def test_extract_text_handles_pandas_empty_data_error(
    tmp_path,
    monkeypatch,
):
    file_path = tmp_path / "test.csv"

    file_path.write_text(
        "dummy",
        encoding="utf-8",
    )

    def mock_read_csv(*args, **kwargs):
        raise pd.errors.EmptyDataError(
            "No columns to parse from file"
        )

    monkeypatch.setattr(
        csv_module.pd,
        "read_csv",
        mock_read_csv,
    )

    with pytest.raises(
        ValueError,
        match="The CSV file is empty.",
    ):
        csv_loader.extract_text(file_path)
        
def test_extract_text_handles_parser_error(
    tmp_path,
    monkeypatch,
):
    file_path = tmp_path / "test.csv"

    file_path.write_text(
        "Name,Age\nAlice,25\n",
        encoding="utf-8",
    )

    def mock_read_csv(*args, **kwargs):
        raise pd.errors.ParserError(
            "Unable to parse CSV"
        )

    monkeypatch.setattr(
        csv_module.pd,
        "read_csv",
        mock_read_csv,
    )

    with pytest.raises(
        ValueError,
        match="The CSV file could not be parsed.",
    ):
        csv_loader.extract_text(file_path)
        
def test_extract_text_handles_os_error(
    tmp_path,
    monkeypatch,
):
    file_path = tmp_path / "test.csv"

    file_path.write_text(
        "Name\nAlice\n",
        encoding="utf-8",
    )

    def mock_read_csv(*args, **kwargs):
        raise OSError("Permission denied")

    monkeypatch.setattr(
        csv_module.pd,
        "read_csv",
        mock_read_csv,
    )

    with pytest.raises(
        ValueError,
        match="Unable to read the CSV file.",
    ):
        csv_loader.extract_text(file_path)
        
def test_extract_text_propagates_unexpected_error(
    tmp_path,
    monkeypatch,
):
    file_path = tmp_path / "test.csv"

    file_path.write_text(
        "Name\nAlice\n",
        encoding="utf-8",
    )

    def mock_read_csv(*args, **kwargs):
        raise RuntimeError(
            "Unexpected CSV failure"
        )

    monkeypatch.setattr(
        csv_module.pd,
        "read_csv",
        mock_read_csv,
    )

    with pytest.raises(
        RuntimeError,
        match="Unexpected CSV failure",
    ):
        csv_loader.extract_text(file_path)
        
def test_extract_text_returns_columns_when_rows_are_empty(
    tmp_path,
):
    file_path = tmp_path / "empty_values.csv"

    file_path.write_text(
        "Name,Age,Role\n"
        ",,\n",
        encoding="utf-8",
    )

    result = csv_loader.extract_text(file_path)

    assert result == "Columns: Name, Age, Role"
        
def test_extract_text_skips_row_with_only_empty_values(
    tmp_path,
):
    file_path = tmp_path / "test.csv"

    file_path.write_text(
        "Name,Age,Role\n"
        ",,\n"
        "Alice,25,Engineer\n",
        encoding="utf-8",
    )

    result = csv_loader.extract_text(file_path)

    assert "Row 1:" not in result
    assert "Row 2:" in result
    assert "Name: Alice" in result
    
def test_extract_text_accepts_path_object(tmp_path):
    file_path = tmp_path / "test.csv"

    file_path.write_text(
        "Name,Age\nAlice,25\n",
        encoding="utf-8",
    )

    result = csv_loader.extract_text(
        Path(file_path)
    )

    assert "Columns: Name, Age" in result
    assert "Name: Alice" in result
    
def test_extract_text_strips_cell_values(tmp_path):
    file_path = tmp_path / "test.csv"

    file_path.write_text(
        "Name,Role\n"
        " Alice , Engineer \n",
        encoding="utf-8",
    )

    result = csv_loader.extract_text(file_path)

    assert "Name: Alice" in result
    assert "Role: Engineer" in result
    
def test_extract_text_rejects_csv_with_no_columns(
    monkeypatch,
    tmp_path,
):
    file_path = tmp_path / "no_columns.csv"

    file_path.write_text(
        "dummy",
        encoding="utf-8",
    )

    dataframe = pd.DataFrame(index=[0])

    monkeypatch.setattr(
        csv_module.pd,
        "read_csv",
        lambda *args, **kwargs: dataframe,
    )

    with pytest.raises(
        ValueError,
        match="The CSV file contains no columns.",
    ):
        csv_loader.extract_text(file_path)
        
def test_extract_text_rejects_no_readable_content(
    monkeypatch,
    tmp_path,
):
    file_path = tmp_path / "unreadable.csv"

    file_path.write_text(
        "Name,Role\n,\n",
        encoding="utf-8",
    )

    dataframe = pd.DataFrame(
        {
            "Name": [""],
            "Role": [""],
        }
    )

    monkeypatch.setattr(
        csv_module.pd,
        "read_csv",
        lambda *args, **kwargs: dataframe,
    )

    with pytest.raises(
        ValueError,
        match="No readable content could be extracted",
    ):
        csv_loader.extract_text(file_path)
        
def test_extract_text_rejects_no_readable_content(
    monkeypatch,
    tmp_path,
):
    file_path = tmp_path / "unreadable.csv"

    file_path.write_text(
        "Name,Role\n,\n",
        encoding="utf-8",
    )

    dataframe = pd.DataFrame(
        {
            "Name": [""],
            "Role": [""],
        }
    )

    monkeypatch.setattr(
        csv_module.pd,
        "read_csv",
        lambda *args, **kwargs: dataframe,
    )

    with pytest.raises(
        ValueError,
        match="No readable content could be extracted",
    ):
        csv_loader.extract_text(file_path)
        
def test_extract_text_handles_raw_text_read_os_error(
    monkeypatch,
    tmp_path,
):
    file_path = tmp_path / "test.csv"

    file_path.write_text(
        "Name,Age\n,\n",
        encoding="utf-8",
    )

    dataframe = pd.DataFrame(
        {
            "Name": [""],
            "Age": [""],
        }
    )

    monkeypatch.setattr(
        csv_module.pd,
        "read_csv",
        lambda *args, **kwargs: dataframe,
    )

    def raise_os_error(self, *args, **kwargs):
        raise OSError(
            "Unable to read raw CSV file"
        )

    monkeypatch.setattr(
        csv_module.Path,
        "read_text",
        raise_os_error,
    )

    with pytest.raises(
        ValueError,
        match="No readable content could be extracted",
    ):
        csv_loader.extract_text(file_path)