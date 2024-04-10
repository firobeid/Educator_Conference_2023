import numpy as np
from tqdm import tqdm
#These functions can be added to the pipline
def calculate_conditions(df_np, missing_threshold):  
    missing = np.mean(np.isnan(df_np), axis=0) > missing_threshold  
    non_variating = np.apply_along_axis(lambda x: len(np.unique(x)) <= 1, axis=0, arr=df_np)  
    return missing, non_variating  
  
def drop_non_informative_columns(df, missing_threshold=0.99):  
    global missing_non_numeric, non_variating_non_numeric, missing, non_variating
    # Separate numeric and non-numeric columns  
    numeric_cols = df.select_dtypes(include=np.number).columns  
    non_numeric_cols = df.select_dtypes(exclude=np.number).columns  
  
    # Convert DataFrame to NumPy array for numeric columns  
    df_numeric_np = df[numeric_cols].values  
  
    # Compute conditions on NumPy array  
    missing, non_variating = calculate_conditions(df_numeric_np, missing_threshold)  
  
    # Columns to drop for numeric columns  
    drop_columns_numeric = numeric_cols[missing | non_variating]  
  
    # Compute conditions for non-numeric columns  
    missing_non_numeric = df[non_numeric_cols].isna().mean() > missing_threshold  
    non_variating_non_numeric = df[non_numeric_cols].nunique() <= 1  
  
    # Columns to drop for non-numeric columns  
    drop_columns_non_numeric = non_numeric_cols[missing_non_numeric | non_variating_non_numeric]  
  
    # Combine all columns to drop  
    drop_columns = list(drop_columns_numeric) + list(drop_columns_non_numeric)  
  
    # Drop columns from DataFrame  
    df.drop(drop_columns, axis=1, inplace=True)  
  
    return df  

def get_uninformative_columns(df, threshold=0.95):  
    """  
    Identify columns from a dataframe that have one category which appears more than 'threshold' proportion.  
  
    :param df: Input dataframe  
    :param threshold: Proportion threshold, default 0.95  
    :return: List of uninformative column names  
    """  
    uninformative_columns = []  
    for column in tqdm(df.select_dtypes(include=['object', 'category']).columns):  
        max_proportion = df[column].value_counts(normalize=True).values[0]  
        if max_proportion > threshold:  
            uninformative_columns.append(column)  
    return uninformative_columns 