"""
Module for EDA preprocessing.
"""
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from .cleaning import *

warnings.simplefilter('ignore')
pd.set_option('display.max_columns', None)


def report(data: pd.DataFrame, nan_threshold: int = -1) -> pd.DataFrame:
    """
    Generates a report of the dataset, including NaN count,
    data types and unique values in the column
    TODO: include more features in the report

    Parameters:
    ----------
    data: pandas.DataFrame
        Dataset
    nan_threshold: int, default=-1

    Returns:
    --------
    report: pandas.DataFrame
        Report structured as a DataFrame
    """
    # nan counting
    cols = [col for col in data.columns if data[col].isna().sum() > nan_threshold]
    nan_counts = data[cols].isna().sum()
    # types
    dtypes = data[cols].dtypes
    # unique values
    uniques = data.nunique()
    report = pd.concat([nan_counts, dtypes, uniques], axis=1)
    report.columns = ['nan_count', 'dtype', 'unique']
    return report


def show_corr_heatmap(data: pd.DataFrame, figsize: tuple, save_figure: bool = False,
                      export_path: str = 'figs/corr_heatmap.png') -> None:
    """
    Creates half correlation heatmap for the dataset. 

    Parameters:
    ----------
    data: pandas.DataFrame
        dataset
    figsize: tuple 
        tuple with dimensions of figure as (x, y) coordinates
    save_figure: bool, default = False
        if True exports the heatmap to export_path
    export_path. str, default = figs/corr_heatmap.png
        Path for the exported heatmap
    """
    mask = np.zeros_like(data.corr())
    mask[np.triu_indices_from(mask)] = True  # upper tridiagonal mask
    plt.subplots(figsize=figsize)
    sns.heatmap(data.corr(), mask=mask, annot=True)
    if save_figure:
        if not os.path.isdir(export_path.split('/')[0]):
            os.makedirs(export_path.split('/')[0])
        plt.savefig(export_path, dpi=600)


def create_multicoll_df(unique_couples, corr):
    """
    Structure multicollinearity information in a DataFrame.
    It should not be called directly.
    """
    data = []
    for couple in unique_couples:
        corr_value = corr[couple[1]].loc[corr.index == couple[0]].item()
        data.append([couple[0], couple[1], corr_value])
    df_mult = pd.DataFrame(data=data, columns=['feat1', 'feat2', 'corr'])
    return df_mult.sort_values(by='feat1').reset_index(drop=True)


def report_multicoll(data: pd.DataFrame, corr_thresh: float = 0.3,
                                                 corr_method='pearson'):
    """
    Identify and report multicollinearity in a dataset.

    Parameters:
    ----------
    data: pandas.DataFrame
        dataset
    corr_thresh: float, default = 0.3
        Correlation threshold that identifies multicollinearity.
    corr_method: {'pearson', 'kendall', 'spearman'} or callable, default='pearson'
        Method of correlation. For additional documentation refer to:
        <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html>

    Returns:
    -------
    
    """
    corr = data.corr(method=corr_method)
    cond = (corr.apply(abs) < corr_thresh) | (corr == 1.0)
    masked_corr = corr.mask(cond)
    masked_corr = masked_corr.dropna(axis=1, how='all')
    masked_corr = masked_corr.dropna(axis=0, how='all')
    multicoll_couples = []
    for feature1 in masked_corr.index:
        feature2_lst = masked_corr.loc[feature1].dropna().index.tolist()
        multicoll_couples.extend([(feature1, feat2) for feat2 in feature2_lst])

    sorted_couples = [sorted(couple) for couple in multicoll_couples]
    unique_couples = [list(x) for x in set(tuple(x) for x in sorted_couples)]
    return create_multicoll_df(unique_couples, corr)


def plot_distributions(data, figsize):
    """
    Plot features distribute. Different distributions can be set for 
    categoricals/numericals variable
    """
    for col in data:
        plt.figure(figsize=figsize)
        if data[col].dtype in ['int64', 'float64']:
            # data is numeric
            sns.distplot(data[col])
        else:
            # data is categorical
            sns.countplot(data[col])


def plot_outliers(data, figsize):
    """
    Plot boxplot of outliers
    """
    for col in data:
        plt.figure(figsize=figsize)
        if data[col].dtype in ['int64', 'float64']:
            # data is numeric
            sns.boxplot(data[col])
        else:
            # data is categorical
            # sns.boxplot(x=col, y=config.TARGET, data=data)
            pass
