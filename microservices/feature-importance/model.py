"""Models used for Feature Importance calculation."""

# Libraries--------------------------------------------------------------------------------------------------

from utils import sort_dictionary_by_value
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import train_test_split 
from sklearn.metrics import matthews_corrcoef, r2_score
import pandas as pd
from pydantic import ValidationError

# Functions--------------------------------------------------------------------------------------------------

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
        stratify=y,
    )

    # Training the model
    model = XGBClassifier()
    model.fit(X_train, y_train)

    # Classification metrics
    y_pred = model.predict(X_test)
    accuracy = matthews_corrcoef(
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
    output['importances'] = sort_dictionary_by_value(output['importances'])

    return output

# Feature Importance for regression models
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
  output['importances'] = sort_dictionary_by_value(output['importances'])

  return output

# Main Class ------------------------------------------------------------------------------------------------
class FeaturesImportancesExtractor():

  def __init__(self, data:pd.DataFrame, target_name:str) -> None:
    """
    Constructor - Takes a dataframe and a target name and returns an object
    
    Args:
      data (pd.DataFrame): The dataframe that contains the data you want to use to train the model.
      target_name (str): The name of the column that contains the target variable.
    
    Returns:
      The object itself.
    """

    # Validate input
    if not isinstance(data, pd.DataFrame):
      raise ValidationError("--data-- must be pandas DataFrame type.")
    
    if not isinstance(target_name, str):
      raise ValidationError("--data-- must be str type.")

    if data.empty:
      raise ValidationError("--data-- can't be an empty dataframe.")

    if len(data.columns) < 2:
      raise ValidationError("--data-- must have at least two columns.")

    if not target_name in data.columns:
      raise ValidationError("--target_name-- must be a column name of --data-- dataframe.")

    # Assign the data
    self.data = data
    self.target_name = target_name

  # Nice representation
  def __str__(self) -> str:
    """
    This function returns a string that contains the shape of the data and the name of the target
    column
    
    Returns:
      The shape of the data and the target column name.
    """
    return f"Data shape: {self.data.shape} -> Target column: {self.target_name}"

  # Full representation
  def __repr__(self) -> str:
    """
    This function returns a string representation of the object
    
    Returns:
      A string representation of the object.
    """
    return f"FeaturesImportancesExtractor(data=pd.DataFrame, target_name={self.target_name})"

  # Feature importances for classifiers
  def clf_feature_importances(self) -> dict:
    """
    It takes in a dataframe and a target column name, and returns a dictionary of feature importances.
    
    Returns:
      A dictionary with the feature importances for classification problem.
    """
    
    output = feature_importance_clf(
      df_input=self.data,
      target_col=self.target_name,
    )
    
    return output

  # Feature importances for regressors
  def reg_feature_importances(self) -> dict:
    """
    It takes in a dataframe, and a target column name, and returns a dictionary of feature
    importances.
    
    Returns:
      A dictionary with the feature importances for each feature.
    """
    
    output = feature_importance_reg(
      df_input=self.data,
      target_col=self.target_name,
    )
    
    return output






