import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
)


def compare_models(
    df,
    target,
    test_size=0.2,
    random_state=42,
):
    """
    Train and compare multiple classification models.
    """

    # =====================================================
    # PREPARE DATA
    # =====================================================

    X = df.drop(columns=[target])
    y = df[target]

    # Remove object columns for now
    X = X.select_dtypes(exclude=["object"])

    # =====================================================
    # TRAIN TEST SPLIT
    # =====================================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    # =====================================================
    # MODELS
    # =====================================================

    models = {
        "DummyClassifier": DummyClassifier(),
        "LogisticRegression": LogisticRegression(
            max_iter=500
        ),
        "RandomForestClassifier": RandomForestClassifier(
            random_state=random_state
        ),
    }

    # =====================================================
    # TRAIN & EVALUATE
    # =====================================================

    results = []

    for name, model in models.items():

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(
            y_test,
            predictions
        )

        f1 = f1_score(
            y_test,
            predictions
        )

        results.append({
            "model": name,
            "accuracy": round(accuracy, 4),
            "f1_score": round(f1, 4),
        })

    # =====================================================
    # RESULTS DATAFRAME
    # =====================================================

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="f1_score",
        ascending=False
    )

    return results_df