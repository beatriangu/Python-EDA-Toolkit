def suggest_visualizations(df, target=None):
    """
    Suggest useful visualizations based on dataset structure.
    """

    suggestions = []

    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    if target:
        suggestions.append("Plot target distribution to inspect balance.")

    if df.isnull().sum().sum() > 0:
        suggestions.append("Plot missing values to inspect data quality.")

    if len(numeric_columns) >= 2:
        suggestions.append("Plot correlation heatmap for numeric features.")

    if len(numeric_columns) > 0:
        suggestions.append("Plot numeric distributions to inspect skewness.")

    if len(categorical_columns) > 0:
        suggestions.append("Plot categorical distributions for text/category variables.")

    return suggestions