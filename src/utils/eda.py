"""
Module for EDA preprocessing.
TODO: ,correlations heatmaps, distribution, outliers, imbalanced, chi2square, variance
"""
from cgi import print_arguments
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from zmq import curve_public
from .cleaning import *
import warnings
warnings.simplefilter('ignore')
pd.set_option('display.max_columns', None)


def report(df: pd.DataFrame, nan_threshold: int = -1) -> pd.DataFrame:
    """
    Generates a report of the dataset, including NaN count,
    data types and unique values in the column
    TODO: include more features in the report

    Parameters:
    df (pandas.DataFrame): dataset
    nan_threshold (int): treshold to filter nans.

    Returns:
    df_report (pandas.DataFrame)
    """
    # nan counting
    cols = [col for col in df.columns if df[col].isna().sum() > nan_threshold]
    nan_counts = df[cols].isna().sum()
    # types
    dtypes = df[cols].dtypes
    # unique values
    uniques = df.nunique()
    df_report = pd.concat([nan_counts, dtypes, uniques], axis=1)
    df_report.columns = ['nan_count', 'dtype', 'unique']
    return df_report


def show_corr_heatmap(df: pd.DataFrame, figsize: tuple, save_figure: bool = False,
                      export_path: str = 'figs/corr_heatmap.png') -> None:
    """
    Creates half correlation heatmap for the dataset. 
    TODO: print couples with highest correlation.

    Parameters:
    df (pandas.DataFrame): dataset
    figsize (tuple): tuple with dimensions of figure
    save_figure (bool): default is False, if True exports the heatmap.
    export_path (str): path for the exported heatmap.
    """
    mask = np.zeros_like(df.corr())
    mask[np.triu_indices_from(mask)] = True  # upper tridiagonal mask
    fig, ax = plt.subplots(figsize=figsize)
    ax = sns.heatmap(df.corr(), mask=mask, annot=True)
    if save_figure:
        if not os.path.isdir(export_path.split('/')[0]):
            os.makedirs(export_path.split('/')[0])
        plt.savefig(export_path, dpi=600)


def create_multicoll_df(unique_couples, corr):
    """Structure multicollinearity information in a DataFrame"""
    data = []
    for couple in unique_couples:
        corr_value = corr[couple[1]].loc[corr.index==couple[0]].item()
        data.append([couple[0], couple[1], corr_value])
    df_mult = pd.DataFrame(data=data, columns=['feat1', 'feat2', 'corr'])
    return df_mult.sort_values(by='feat1').reset_index(drop=True)

def report_multicoll(df: pd.DataFrame, corr_thresh: float = 0.3, corr_method='pearson'):
    """Handles multicollinearity in the problem. """
    corr = df.corr(method=corr_method)
    cond = (corr.apply(abs) < corr_thresh) | (corr == 1.0)
    masked_corr = corr.mask(cond)
    masked_corr = masked_corr.dropna(axis=1, how='all')
    masked_corr = masked_corr.dropna(axis=0, how='all')
    multicoll_couples = []
    for feature1 in masked_corr.index:
        feature2_lst = masked_corr.loc[feature1].dropna().index.tolist()
        multicoll_couples.extend([(feature1, feat2) for feat2 in feature2_lst])
    
    sorted_couples = [sorted(couple) for couple in multicoll_couples]
    unique_couples =  [list(x) for x in set(tuple(x) for x in sorted_couples)]
    return create_multicoll_df(unique_couples, corr)

def test():
    """Test the module"""
    df = load_data(config.cleaned_data_path)
    df_mult = report_multicoll(df)
    print(df_mult)


if __name__ == '__main__':
    test()
