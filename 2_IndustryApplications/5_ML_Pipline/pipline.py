import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline    
from typing import Tuple  
from sklearn.impute import SimpleImputer    
from sklearn.preprocessing import StandardScaler    
from sklearn.compose import ColumnTransformer    
from sklearn.model_selection import train_test_split    
from xgboost import XGBClassifier    
import category_encoders as ce   
from sklearn.base import TransformerMixin  
from dataclasses import dataclass  
from sklearn.base import BaseEstimator, ClassifierMixin      
import gc  
import xgboost as xgb    
from sklearn.metrics import roc_curve    

# Load the dataset  
df = pd.read_csv('train.csv')  
# df.head()  
# df.info()  
# df.describe()  
# df.isnull().sum()

# Setup X and y variables
X = df.drop(columns='y')
y = df['y'].values.reshape(-1,1)
y = np.where(y == 'no', 0, 1)

# Split the data into training and testing sets
X_train, X_holdout, y_train, y_holdout = train_test_split(X, y, random_state=42)

numerical_cols = X.select_dtypes(include=np.number).columns.tolist()  
categorical_cols = X.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()  

class ColumnNameKeeper(TransformerMixin):  
    def fit(self, X, y=None):  
        return self  
  
    def transform(self, X):  
        self.column_names = X.columns  
        return X  
    
class NullImputa(TransformerMixin):   
    '''
    __Author__  = Firad Obeid
    '''   
    def __init__(self, min_count_na = 5):      
        self.min_count_na = min_count_na      
        self.missing_cols = None    
        self.additional_features = []  # Store names of additional features  
        self.column_names = None  # Store column names after transformation  
        
    def fit(self, X, y=None):      
        self.missing_cols = X.columns[X.isnull().any()].tolist()      
        return self      
        
    def transform(self, X, y=None):      
        X = X.copy()  # create a copy of the input DataFrame      
        for col in X.columns.tolist():      
            if col in X.columns[X.dtypes == object].tolist():      
                X[col] = X[col].fillna(X[col].mode()[0])      
            else:      
                if col in self.missing_cols:      
                    new_col_name = f'{col}-mi'  
                    X[new_col_name] = X[col].isna().astype(int)     
                    self.additional_features.append(new_col_name)  # Store the new column name  
                if X[col].isnull().sum() <= self.min_count_na:      
                   X[col] = X[col].fillna(X[col].median())      
                else:      
                    X[col] = X[col].fillna(-9999)      
        assert X.isna().sum().sum() == 0      
        _ = gc.collect()      
        print("Imputation complete.....") 
        self.column_names = X.columns.tolist()  # Store column names after transformation  
        return X  

    
def ks_stat(y_pred, dtrain)-> Tuple[str, float]:
    y_true = dtrain.get_label()  
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)  
    ks_stat = max(tpr - fpr)  
    return 'ks_stat', ks_stat 

class XGBoostClassifierWithEarlyStopping(BaseEstimator, ClassifierMixin):    
    def __init__(self, nfolds=5, **params):    
        self.params = params    
        self.evals_result = {}    
        self.bst = None  
        self.nfolds = nfolds  
        self.cvresult = None  # initialize cvresult attribute 
    
    def fit(self, X, y, **fit_params):    
        dtrain = xgb.DMatrix(X, label=y)  
        self.cvresult = xgb.cv(self.params, dtrain, num_boost_round=10000,  verbose_eval=True, maximize=True,
                          nfold=self.nfolds, metrics=['auc'], custom_metric = ks_stat,
                          early_stopping_rounds=10, stratified=True,  
                          seed=42)  
        self.bst = xgb.train(self.params, dtrain, num_boost_round=self.cvresult.shape[0], feval = ks_stat)  
        return self    
    
    def predict(self, X):    
        dtest = xgb.DMatrix(X)    
        return self.bst.predict(dtest)  

  
def init_model_params(y_train):
    '''
    “binary:logistic” –logistic regression for binary classification, output probability
    “binary:logitraw” –logistic regression for binary classification, output score before logistic transformation
    '''
    #define class weight dictionary, negative class has 20x weight
    # w = {0:20, 1:1}
    global params
    global plst
    # plst = 
    y_train = pd.DataFrame(y_train)
    params = {"booster" :"gbtree",
              "max_depth" : 5,
              "n_jobs": -1,
              "verbosity" : 3,
             "objective": "binary:logistic",
             "eta": 0.05,
              "colsample_bytree" : 0.3, 
             "tree_method": "exact",
             "scale_pos_weight": int((y_train.value_counts()[0] / y_train.value_counts()[1])),
             "eval_metric": ["auc", "logloss", "error"],
             "subsample" : 0.8, "colsample_bylevel" : 1, "random_state" : 42, "verbosity" : 3}   

init_model_params(y_train)
# Create a preprocessor for numerical columns  
numeric_transformer = Pipeline(steps=[  
    ('custome-imputer', NullImputa(5)),  
    ('scaler', StandardScaler())])  

# Create a preprocessor for categorical columns  
categorical_transformer = Pipeline(steps=[  
    ('custome-imputer', NullImputa(5)),  
    ('target_encoder', ce.TargetEncoder())])  
 

# Combine the preprocessors using a ColumnTransformer  
preprocessor = ColumnTransformer(  
    transformers=[  
        ('num', numeric_transformer, numerical_cols),  
        ('cat', categorical_transformer, categorical_cols)])  
  
# Create a pipeline that combines the preprocessor with the estimator  
pipeline = Pipeline(steps=[('preprocessor', preprocessor),  
                           ('classifier', XGBoostClassifierWithEarlyStopping(**params))])  
  
# Fit the pipeline to the training data  
pipeline.fit(X_train, y_train)  
# classifier__eval_set=[(pipeline.named_steps['preprocessor'].transform(X_test), y_test)])
# Now you can use pipeline.predict() to make predictions on new data 
# pipeline.predict(X_holdout) 

cv_results = pipeline.named_steps['classifier'].cvresult
cv_results[cv_results.columns[cv_results.columns.str.contains('ks.*mean|mean.*ks', regex=True)]].plot()

y_train_pred = pipeline.predict(X_train) 
y_holdout_pred = pipeline.predict(X_holdout) 

# Fit and transform the data with preprocessor  
# preprocessed_X_train = pipeline.named_steps['preprocessor'].transform(X_train)  
# Access the underlying Booster in XGBoost   
importances = pipeline.named_steps['classifier'].bst.get_score(importance_type='gain')  

preprocessor = pipeline.named_steps['preprocessor']  

numeric_column_names_after_preprocessing = preprocessor.transformers_[0][1].named_steps['custome-imputer'].column_names  
categorical_column_names_after_preprocessing = preprocessor.transformers_[1][1].named_steps['custome-imputer'].column_names  

features_after_preprocessing = numeric_column_names_after_preprocessing + categorical_column_names_after_preprocessing

# Map to actual feature names  
importances_with_feat_names = {features_after_preprocessing[int(feat[1:])]: imp for feat, imp in importances.items()}  
print("Number of features after preprocessing:", len(features_after_preprocessing))  
print("Max feature index in importances:", max(int(feat[1:]) for feat in importances.keys()))

importances = pd.DataFrame(list(importances_with_feat_names.items()), columns=['Feature', 'Importance'])  
importances['Normalized_Importance'] = importances['Importance'] / importances['Importance'].sum()
# Sort the DataFrame by importance  
importances = importances.sort_values(by='Normalized_Importance', ascending=False)  