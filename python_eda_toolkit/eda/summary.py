import pandas as pd


def resumen_dataset(df):
    """
    Devuelve información básica del DataFrame.
    """
    return {
        "filas": df.shape[0],
        "columnas": df.shape[1],
        "duplicados": df.duplicated().sum(),
        "valores_nulos": df.isnull().sum().sum(),
    }


def valores_nulos(df):
    """
    Devuelve columnas con valores nulos.
    """
    nulos = df.isnull().sum()
    return nulos[nulos > 0].sort_values(ascending=False)


def tipos_columnas(df):
    """
    Devuelve los tipos de datos de cada columna.
    """
    return pd.DataFrame({
        "columna": df.columns,
        "tipo": df.dtypes.values
    })


def resumen_categoricas(df):
    """
    Devuelve un resumen de variables categóricas.
    """
    categoricas = df.select_dtypes(include=["object", "category"])

    resumen = {}

    for col in categoricas.columns:
        resumen[col] = {
            "valores_unicos": categoricas[col].nunique(),
            "valores": categoricas[col].unique().tolist()
        }

    return resumen
