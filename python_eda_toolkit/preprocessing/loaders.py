from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


SUPPORTED_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".txt",
    ".xlsx",
    ".xls",
    ".json",
    ".jsonl",
    ".parquet",
    ".pq",
    ".feather",
    ".pkl",
    ".pickle",
}


def _validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate that the loaded object is a usable pandas DataFrame.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            "Loaded data must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            "Loaded dataset is empty."
        )

    if df.shape[1] == 0:
        raise ValueError(
            "Loaded dataset has no columns."
        )

    return df


def _normalize_path(path: str | Path) -> Path:
    """
    Convert a string path into a Path object and validate existence.
    """

    file_path = Path(path).expanduser()

    if not file_path.exists():
        raise FileNotFoundError(
            f"File not found: {file_path}"
        )

    if not file_path.is_file():
        raise ValueError(
            f"Path is not a file: {file_path}"
        )

    return file_path


def _read_csv_like(
    file_path: Path,
    sample_size: int | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Read CSV, TSV or TXT files with flexible separator detection.
    """

    extension = file_path.suffix.lower()

    default_kwargs = {
        "encoding": "utf-8",
    }

    if extension == ".tsv":
        default_kwargs["sep"] = "\t"
    else:
        default_kwargs["sep"] = None
        default_kwargs["engine"] = "python"

    default_kwargs.update(kwargs)

    try:
        df = pd.read_csv(
            file_path,
            nrows=sample_size,
            **default_kwargs,
        )

    except UnicodeDecodeError:
        fallback_kwargs = default_kwargs.copy()
        fallback_kwargs["encoding"] = "latin1"

        df = pd.read_csv(
            file_path,
            nrows=sample_size,
            **fallback_kwargs,
        )

    return df


def _read_excel(
    file_path: Path,
    sheet_name: str | int | None = 0,
    sample_size: int | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Read Excel files.
    """

    df = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        **kwargs,
    )

    if isinstance(df, dict):
        first_sheet = next(iter(df))
        df = df[first_sheet]

    if sample_size is not None:
        df = df.head(sample_size)

    return df


def _read_json(
    file_path: Path,
    sample_size: int | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Read JSON or JSON Lines files.
    """

    extension = file_path.suffix.lower()

    if extension == ".jsonl":
        df = pd.read_json(
            file_path,
            lines=True,
            **kwargs,
        )
    else:
        try:
            df = pd.read_json(
                file_path,
                **kwargs,
            )
        except ValueError:
            df = pd.read_json(
                file_path,
                lines=True,
                **kwargs,
            )

    if sample_size is not None:
        df = df.head(sample_size)

    return df


def _read_parquet(
    file_path: Path,
    sample_size: int | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Read Parquet files.
    """

    df = pd.read_parquet(
        file_path,
        **kwargs,
    )

    if sample_size is not None:
        df = df.head(sample_size)

    return df


def _read_feather(
    file_path: Path,
    sample_size: int | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Read Feather files.
    """

    df = pd.read_feather(
        file_path,
        **kwargs,
    )

    if sample_size is not None:
        df = df.head(sample_size)

    return df


def _read_pickle(
    file_path: Path,
    sample_size: int | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Read Pickle files.
    """

    df = pd.read_pickle(
        file_path,
        **kwargs,
    )

    if sample_size is not None:
        df = df.head(sample_size)

    return df


def load_data(
    data: str | Path | pd.DataFrame,
    sample_size: int | None = None,
    sheet_name: str | int | None = 0,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Load data into a pandas DataFrame.

    Supported inputs:
    - pandas DataFrame
    - CSV
    - TSV
    - TXT
    - Excel
    - JSON
    - JSON Lines
    - Parquet
    - Feather
    - Pickle

    Parameters
    ----------
    data : str, Path or pandas.DataFrame
        Dataset source.

    sample_size : int, optional
        Number of rows to load. Useful for large datasets.

    sheet_name : str, int or None, default=0
        Excel sheet to load.

    **kwargs
        Extra keyword arguments passed to the pandas reader.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """

    if isinstance(data, pd.DataFrame):
        df = data.copy()

        if sample_size is not None:
            df = df.head(sample_size)

        return _validate_dataframe(df)

    if not isinstance(data, (str, Path)):
        raise TypeError(
            "Data must be a file path or a pandas DataFrame."
        )

    file_path = _normalize_path(data)
    extension = file_path.suffix.lower()

    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file format '{extension}'. "
            f"Supported formats are: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    try:
        if extension in {".csv", ".tsv", ".txt"}:
            df = _read_csv_like(
                file_path=file_path,
                sample_size=sample_size,
                **kwargs,
            )

        elif extension in {".xlsx", ".xls"}:
            df = _read_excel(
                file_path=file_path,
                sheet_name=sheet_name,
                sample_size=sample_size,
                **kwargs,
            )

        elif extension in {".json", ".jsonl"}:
            df = _read_json(
                file_path=file_path,
                sample_size=sample_size,
                **kwargs,
            )

        elif extension in {".parquet", ".pq"}:
            df = _read_parquet(
                file_path=file_path,
                sample_size=sample_size,
                **kwargs,
            )

        elif extension == ".feather":
            df = _read_feather(
                file_path=file_path,
                sample_size=sample_size,
                **kwargs,
            )

        elif extension in {".pkl", ".pickle"}:
            df = _read_pickle(
                file_path=file_path,
                sample_size=sample_size,
                **kwargs,
            )

        else:
            raise ValueError(
                f"Unsupported file format: {extension}"
            )

    except Exception as error:
        raise RuntimeError(
            f"Could not load file '{file_path}'. "
            f"Original error: {error}"
        ) from error

    return _validate_dataframe(df)