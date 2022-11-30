"""Models used for Feature Importance calculation."""

# Libraries
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, r2_score
import pandas as pd

# Feature Importance for classification models
def feature_importance_clf(df_input: pd.DataFrame, target_col: str) -> dict:
    """
    It takes a dataframe and a target column as input, and returns a dictionary with the model accuracy
    and the feature importances.
    
    Args:
      df_input (pd.DataFrame): pd.DataFrame - the dataset you want to use to train the model
      target_col (str): The name of the column that contains the target variable.
    
    Returns:
      A dictionary with the model accuracy and the feature importances.
    """

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
    accuracy = accuracy_score(
        y_test, 
        y_pred,
    )
    
    output['model accuracy'] = accuracy

    # Feature importances
    output['importances'] = {}
    attributes = X_train.columns
    importances = model.feature_importances_

    for att, imp in zip(attributes, importances):
        output['importances'][att] = imp

    # Sort output
    output['importances'] = {
        key: val for key, val in sorted(
            output['importances'].items(), 
            key=lambda item: item[1],
        )
    }

    return output

def feature_importance_reg(df_input: pd.DataFrame, target_col: str) -> dict:
    """
    It takes a dataframe and a target column, splits the dataframe into training and test sets, trains a
    model, and returns the model's r2 score and feature importances
    
    Args:
      df_input (pd.DataFrame): pd.DataFrame - The dataset you want to use to train the model.
      target_col (str): The name of the column that contains the target variable.
    
    Returns:
      A dictionary with the r2 score and a dictionary with the feature importances.
    """

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
    )

    # Training the model
    model = XGBRegressor()
    model.fit(X_train, y_train)

    # Classification metrics
    y_pred = model.predict(X_test)
    r2 = r2_score(
        y_test, 
        y_pred,
    )
    
    output['r2'] = r2

    # Feature importances
    output['importances'] = {}
    attributes = X_train.columns
    importances = model.feature_importances_

    for att, imp in zip(attributes, importances):
        output['importances'][att] = imp

    # Sort output
    output['importances'] = {
        key: val for key, val in sorted(
            output['importances'].items(), 
            key=lambda item: item[1],
        )
    }

    return output



