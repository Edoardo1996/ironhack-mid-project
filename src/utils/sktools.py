"""
Module for applying and scoring a ML model for a regression or classification problem.
It defines a pipeline where outliers, encoders, scalers and ML model can be tuned.
"""
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, \
    confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsRegressor


def remove_outliers(df: pd.DataFrame, skip_columns: list[str],
                    threshold: float = 1.5, verbose: bool = False) -> pd.DataFrame:
    """
    Removes outliers from the dataset.

    Parameters:
    df (pandas.DataFrame): dataset
    skip_columns (list[str]): Columns for which outliers are not removed
    threshold (float): Threshold for removal of outliers
    verbose (bool): Default False, if True print details of removal of outliers

    Returns:
    df (pandas.DataFrame): dataset without outliers
    """
    initial_size = len(df)
    for col in df.select_dtypes(np.number).columns:
        if col not in skip_columns:
            upper = np.percentile(df[col], 75)
            lower = np.percentile(df[col], 25)
            iqr = upper - lower
            upper_limit = upper + threshold * iqr
            lower_limit = lower - threshold * iqr
            df = df[(df[col] > lower_limit) & (df[col] < upper_limit)]
            assert not df.empty, 'Threshold too high for col: ' + col
    if verbose:
        print('Outliers removal has removed {} rows ({} % of initial size)'.format(
            initial_size-len(df), round((1-len(df)/initial_size)*100, 2)
        ))
    return df


def split_data(df: pd.date_range, target: str, test_size: float,
               random_state: int):
    """
    Split the dataset into random train and test subset

    Parameters:
    df (pandas.DataFrame): dataset
    target (str): column that denotes the target for predictions
    test_size (float): size of the training subset
    random_state (int): Controls the shuffling applied to the data before the split.

    Returns:
    X_train (np.array): Data of training features
    X_test (np.array): Data of test features
    y_train (np.array): Data of train target
    y_test (np.array): Data of test target
    """
    X = df.drop([target], axis=1)
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test


def scale_data(X_train: np.array, X_test: np.array, scaler_class) -> None:
    """
    Fit chosen scaler class from scikit-learn on training data and transforms
    both training and test data.

    Parameters:
    X_train (np.array): Data of training features
    X_test (np.array): Data of test features
    scaler_class (class): Class of scikit-learn for scaling the data
    """
    scaler = scaler_class.fit(X_train.select_dtypes(np.number))
    X_train_scaled = scaler.transform(X_train.select_dtypes(np.number))
    X_train[X_train.select_dtypes(np.number).columns] = X_train_scaled
    X_test_scaled = scaler.transform(X_test.select_dtypes(np.number))
    X_test[X_test.select_dtypes(np.number).columns] = X_test_scaled


def encode_data(X_train: np.array, X_test: np.array, encoders: list,
                cols_to_encode: list[str]):
    """
    Fit chosen encoding class from scikit-learn on training data and transforms
    both training and test data.

    Parameters:
    X_train (np.array): Data of training features
    X_test (np.array): Data of test features
    encoders (list): List of encoders to apply to corresponding cols_to_encode
    cols_to_encode(list[str]): list of columns in which apply the encoding

    Returns:
    X_train (np.array): Data of training features encoded
    X_test (np.array): Data of test features encoded
    """
    for encoder, cols in zip(encoders, cols_to_encode):  # loop on chosen encoders and columns
        if (isinstance(encoder, OrdinalEncoder)) and bool(list(cols)):
            for col in cols:
                encoder_model = encoder.fit(X_train[col].values.reshape(-1, 1))
                X_train[col] = encoder_model.transform(
                    X_train[col].values.reshape(-1, 1))
                X_test[col] = encoder_model.transform(
                    X_test[col].values.reshape(-1, 1))
        elif isinstance(encoder, OneHotEncoder) and bool(list(cols)):
            encoder_model = encoder.fit(X_train[cols])
            X_train_onehot_encoded = pd.DataFrame(
                encoder_model.transform(X_train[cols]).toarray())
            X_train_onehot_encoded = X_train_onehot_encoded.set_index(
                X_train.index)
            X_train = X_train.drop(X_train[cols], axis=1).join(
                X_train_onehot_encoded)
            X_test_onehot_encoded = pd.DataFrame(
                encoder_model.transform(X_test[cols]).toarray())
            X_test_onehot_encoded = X_test_onehot_encoded.set_index(
                X_test.index)
            X_test = X_test.drop(X_test[cols], axis=1).join(
                X_test_onehot_encoded)
        elif bool(list(cols)):
            # Columns are present but no encoder was recognized
            sys.exit("Encoder not recognized, please use another")

    return X_train, X_test


def apply_model(X_train: np.array, X_test: np.array, y_train: np.array, model,
                return_formula: bool) -> np.array:
    """
    Apply a ML model to a scaled and encoded dataset
    """
    model.fit(X_train, y_train)
    if return_formula:
        print('Coefficients:')
        print(model.coef_, end='\n\n')
        print('Intercept:')
        print(model.intercept_, end='\n\n')

    return model.predict(X_test.dropna())


def save_results(path, results, append=True, variable=None):
    """
    Save results and metrics in a file for further manipulation
    Data is in format: r2, mae, mse
    TODO: function should save in a file user-defined metrics too.
    """
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


def score_regression_model(df: pd.DataFrame, target: str, model,
                           return_formula=False,
                           cols_to_encode=None,
                           scaler=None, encoders=None,
                           cols_to_drop=[],
                           test_size=0.3, random_state=42,
                           outsiders_thresh=None,
                           skip_outsiders_cols=[]):
    """
    Scores a Regression Model, it assumes data is already cleaned

    Parameters:
    df (pd.DataFrame): Dataset for our model
    target (str): Name of target column
    outsiders_thresh (float): Threshold for the removal of outliers
    scaler (class): Scaling method for numerical data
    encoder (list): Encoding methods for categorical data
    cols_to_encode (list): Columns for encoding methods
    model (class): ML model for regression

    Returns:
    predictions (np.array): predicted target values
    r2 (float): r2 score of the method
    mae(float): mean absolute error 
    mse(float): mean squared error
    """
    df = df.drop(cols_to_drop, axis=1)
    if outsiders_thresh:
        df = remove_outliers(df,
                             threshold=outsiders_thresh,
                             skip_columns=skip_outsiders_cols + [target])
    X_train, X_test, y_train, y_test = split_data(
        df, target, test_size, random_state)
    if scaler:
        scale_data(X_train, X_test, scaler)
    if encoders:
        X_train, X_test = encode_data(
            X_train, X_test, encoders, cols_to_encode)
    predictions = apply_model(X_train, X_test, y_train, model, return_formula)

    return (predictions,
            r2_score(y_test, predictions),
            mean_absolute_error(y_test, predictions),
            mean_squared_error(y_test, predictions, squared=False))


def score_classification_model(df, target,
                               model,
                               return_formula=False,
                               cols_to_encode=None,
                               scaler=None, encoders=None,
                               cols_to_drop=[],
                               test_size=0.3, random_state=42,
                               outsiders_thresh=None,
                               skip_outsiders_cols=[]):
    """
    Scores a Classification Model, it assumes data is already cleaned

    Parameters:
    df (pd.DataFrame): Dataset for our model
    target (str): Name of target column
    outsiders_thresh (float): Threshold for the outliers
    scaler (class): Scaling method for numerical data
    encoder (list): Encoding methods for categorical data
    cols_to_encode (list): Columns for encoding methods
    model (class): ML model for classification


    Returns:
    predictions (np.array): predicted target values
    classification_report: metrics for evaluation of the model
    """
    df = df.drop(cols_to_drop, axis=1)
    if outsiders_thresh:
        df = remove_outliers(df,
                             threshold=outsiders_thresh,
                             skip_columns=skip_outsiders_cols + [target])
    X_train, X_test, y_train, y_test = split_data(
        df, target, test_size, random_state)
    if scaler:
        scale_data(X_train, X_test, scaler)
    if encoders:
        X_train, X_test = encode_data(
            X_train, X_test, encoders, cols_to_encode)
    predictions = apply_model(X_train, X_test, y_train, model, return_formula)

    return predictions, classification_report(y_test, predictions)


def knn_optimization(X_train, y_train, X_test, y_test, metric, k, show_plot=True):
    """Try to find a optimal k for the KNNregression algoritmh

    Parameters:
    metric(func): chosen metric for which k is studied 
    k(list): list of chosen k for optimization

    Returns:
    """
    # Plot error rates
    rate = []  # list of metric calculations
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
