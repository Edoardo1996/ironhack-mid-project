"""
Module with simple wrapper functions for classes and methods for EDA or analysis
from scikit-learn
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsRegressor

# more in depth data exploration
def report(df: pd.DataFrame, nan_threshold=-1) -> pd.DataFrame:
    """Report function, to be improved with other indicators in future"""
    # nan counting
    cols = [col for col in df.columns if df[col].isna().sum() > nan_threshold]
    nan_counts = df[cols].isna().sum()
    # types
    dtypes = df[cols].dtypes
    # unique values
    uniques = df.nunique()
    df_nan = pd.concat([nan_counts, dtypes, uniques], axis=1)
    df_nan.columns = ['nan_count', 'dtype', 'unique']
    return df_nan

# functions for implementation of linear regression methods
def show_corr_heatmap(df, figsize, export_path='figs/corr_heatmap.png'):
    """Export half correlation matrix for the dataset in a .png figure 
    TODO: print couples with highest correlation"""
    mask = np.zeros_like(df.corr()) # Matrix full of zeros with shape of df.corr()
    mask[np.triu_indices_from(mask)] = True # upper tridiagonal mask
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(df.corr(), mask=mask, annot=True)
    if not os.path.isdir(export_path.split('/')[0]):
        os.makedirs(export_path.split('/')[0])
    plt.savefig(export_path, dpi=600)
        

def remove_outliers(df, skip_columns, threshold=1.5):
    initial_size = len(df)
    for col in df.select_dtypes(np.number).columns:
        if col not in skip_columns:
            upper = np.percentile(df[col], 75)  
            lower = np.percentile(df[col], 25)  
            iqr = upper - lower
            upper_limit = upper + threshold * iqr
            lower_limit = lower - threshold * iqr
            df = df[(df[col]>lower_limit) & (df[col]<upper_limit)]
            assert not df.empty, 'Threshold too high for col: ' + col
    print('Outliers removal has removed {} rows ({} % of initial size)'.format(
        initial_size-len(df), round((1-len(df)/initial_size )*100, 2)
    ))
    return df

def split_data(df, target, test_size, random_state):
    """Separes the features from target and """
    X = df.drop([target], axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                        test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

def scale_data(X_train, X_test, scaler_class):
    """Fit chosen sclaer class on training data and transforms bot training and test data"""
    scaler = scaler_class.fit(X_train.select_dtypes(np.number))
    X_train_scaled = scaler.transform(X_train.select_dtypes(np.number))
    X_train[X_train.select_dtypes(np.number).columns] = X_train_scaled
    X_test_scaled = scaler.transform(X_test.select_dtypes(np.number))
    X_test[X_test.select_dtypes(np.number).columns] = X_test_scaled
    
def encode_data(X_train, X_test, encoders, cols_to_encode):
    """Apply a encoder class to categorical data"""
    for encoder, cols in zip(encoders, cols_to_encode): # loop on chosen encoders and columns
        if (isinstance(encoder, OrdinalEncoder)) and bool(list(cols)):
            for col in cols:
                encoder_model = encoder.fit(X_train[col].values.reshape(-1,1))
                X_train[col] = encoder_model.transform(X_train[col].values.reshape(-1,1))
                X_test[col] = encoder_model.transform(X_test[col].values.reshape(-1, 1))
        elif isinstance(encoder, OneHotEncoder) and bool(list(cols)):
            encoder_model = encoder.fit(X_train[cols])
            X_train_onehot_encoded = pd.DataFrame(encoder_model.transform(X_train[cols]).toarray())
            X_train_onehot_encoded = X_train_onehot_encoded.set_index(X_train.index) 
            X_train = X_train.drop(X_train[cols], axis=1).join(X_train_onehot_encoded)
            X_test_onehot_encoded = pd.DataFrame(encoder_model.transform(X_test[cols]).toarray())
            X_test_onehot_encoded = X_test_onehot_encoded.set_index(X_test.index) 
            X_test = X_test.drop(X_test[cols], axis=1).join(X_test_onehot_encoded)
        elif bool(list(cols)):
            # Columns are present but no encoder was recognized
            sys.exit("Encoder not recognized, please use another")

    return X_train, X_test

def apply_model(X_train, X_test, y_train, model, return_formula):
    model.fit(X_train, y_train)
    if return_formula:
        print('Coefficients:')
        print(model.coef_, end='\n\n')
        print('Intercept:')
        print(model.intercept_, end='\n\n')
    
    return model.predict(X_test.dropna())

def score_method(df, target, 
                 model,
                 return_formula=False,
                 cols_to_encode=None,
                 scaler=None, encoders=None,
                 cols_to_drop=[],
                 test_size=0.3, random_state=42,
                 outsiders_thresh=None,
                 skip_outsiders_cols = []):
    """
    Scores a Linear Regression Model, it assumes data is already cleaned

    Parameters:
    df (pd.DataFrame): Dataset for our model
    target (str): Name of target column
    outsiders_thresh (float): Threshold for the outliers
    outsiders_cols (list): List of columns in which filter out the outliers
    scaler (class): Scaling method for numerical data
    encoder (list): Encoding methods for categorical data
    cols_to_encode (list): Columns for encoding methods
    model (class): ML model
    

    Returns:
    r2 (float): r2 score of the method
    mae(float): mean absolute error 
    mse(float): mean squared error
    """
    df = df.drop(cols_to_drop, axis=1)
    if outsiders_thresh:
        df = remove_outliers(df,
                            threshold=outsiders_thresh,
                            skip_columns=skip_outsiders_cols + [target])
    X_train, X_test, y_train, y_test = split_data(df, target, test_size, random_state)
    if scaler:
        scale_data(X_train, X_test, scaler)
    if encoders:
        X_train, X_test = encode_data(X_train, X_test, encoders, cols_to_encode)
    predictions = apply_model(X_train, X_test, y_train, model, return_formula)

    return (predictions,
           r2_score(y_test, predictions),
           mean_absolute_error(y_test, predictions),
           mean_squared_error(y_test, predictions, squared=False))

def save_results(path, results, append=True, variable=None):
    """Save results in a file
    Data is in format: r2, mae, mse"""
    if not os.path.isdir(path.split('/')[0]):
        os.makedirs(path.split('/')[0])
    if append:
        f = open(path, 'a')
    else:
        f = open(path, 'w')
    f.write(str(results).replace('(', '').replace(')', ''))
    if variable:
        f.write(' ' + str(variable) + '\n')
    else:
        f.write(' ' + str(variable) + '\n')
    f.close()

def knn_optimization(X_train, y_train, X_test, y_test, metric, k, show_plot=True):
    """Try to find a optimal k for the KNNregression algoritmh

    Parameters:
    metric(func): chosen metric for which k is studied 
    k(list): list of chosen k for optimization

    Returns:
    """
    # Plot error rates 
    rate = [] # list of metric calculations
    for i in range(1, max(k)):
        knn = KNeighborsRegressor(n_neighbors=i)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        rate.append(metric(y_test, y_pred))

    if show_plot:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, max(k)), rate, color='blue', linestyle='dashed',
                marker='o', markerfacecolor='red', markersize=10)
        plt.title(metric.__name__ + ' vs. K Value')
        plt.xlabel('K')
        plt.ylabel(metric.__name__)