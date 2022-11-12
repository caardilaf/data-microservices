"""Models used for Feature Importance calculation."""

# Libraries
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report
import pandas as pd

# Feature Importance for classification models
def feature_importance_clf(df_input: pd.DataFrame, target_col: str) -> dict:

    # Output
    output = {}

    # Dataset values
    y = df_input[target_col]
    X = df_input.drop(target_col, axis=1)

    # Training and tests datasets
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=0.20, 
        random_state=42,
        stratify=y
    )

    # Training the model
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # Classification metrics
    y_pred = model.predict(X_test)
    metrics = classification_report(
        y_test, 
        y_pred,
        output_dict=True
    )

    # Feature importances
    attributes = X_train.columns
    importances = model.feature_importances_

    for att, imp in zip(attributes, importances):
        output[att] = imp

    return output




