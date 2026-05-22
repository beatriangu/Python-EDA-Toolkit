from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def separar_x_y(df, target):
    """
    Separa variables predictoras X y variable objetivo y.
    """
    X = df.drop(columns=[target])
    y = df[target]

    return X, y


def dividir_train_test(X, y, test_size=0.2, random_state=42, stratify=False):
    """
    Divide datos en entrenamiento y prueba.
    """
    stratify_param = y if stratify else None

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )


def escalar_train_test(X_train, X_test):
    """
    Escala X_train y X_test usando StandardScaler.
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


def label_encode_columna(df, columna):
    """
    Codifica una columna categórica con LabelEncoder.
    """
    encoder = LabelEncoder()
    df[columna] = encoder.fit_transform(df[columna])

    return df, encoder
