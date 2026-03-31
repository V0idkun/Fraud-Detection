import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

def load_and_split(path):
    df = pd.read_csv(path)
    # BAF specific: Train on months 0-5, Test on 6-7
    train = df[df['month'] < 6]
    test = df[df['month'] >= 6]
    
    X_train = train.drop('fraud_bool', axis=1)
    y_train = train['fraud_bool']
    X_test = test.drop('fraud_bool', axis=1)
    y_test = test['fraud_bool']
    
    return X_train, X_test, y_train, y_test

def apply_smote(X_train, y_train):
    """
    Only apply SMOTE to the training data to balance the classes.
    """
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    return X_res, y_res

def get_preprocessor(X):
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    num_cols = X.select_dtypes(exclude=['object']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
        ])
    return preprocessor