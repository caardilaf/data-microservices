# Libraries and modules
from model import FeaturesImportancesExtractor
import pandas as pd
from sklearn.datasets import load_iris, load_diabetes

# Let's debug
if __name__ == "__main__":

    # Data de prueba
    data = load_iris()
    X = data['data']
    y = data['target']
    df = pd.concat([pd.DataFrame(X), pd.DataFrame(y)], axis=1)
    columns = [f'f{x}' for x in range(1, df.shape[1])]
    columns.append('target')
    df.columns = columns

    # Probando la clase
    feature_extractor = FeaturesImportancesExtractor(data=df, target_name='target')
    importances = feature_extractor.clf_feature_importances()

    print(importances)
