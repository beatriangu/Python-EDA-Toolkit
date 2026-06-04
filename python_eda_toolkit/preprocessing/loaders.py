import pandas as pd


def load_data(data):
    """
    Load data from:
    - pandas DataFrame
    - CSV file
    - Excel file

    Parameters
    ----------
    data : str or pandas.DataFrame
        Path to the file or a DataFrame.

    Returns
    -------
    pandas.DataFrame
        Loaded dataset.
    """

    # If data is a file path
    if isinstance(data, str):

        # CSV files
        if data.endswith(".csv"):
            return pd.read_csv(
                data,
                sep=None,
                engine="python"
            )

        # Excel files
        if data.endswith((".xlsx", ".xls")):
            return pd.read_excel(data)

        raise ValueError(
            "Unsupported format. Use CSV, Excel or a DataFrame."
        )

    # If data is already a DataFrame
    return data