"""
Functions for loading and cleaning the dataset
"""

from importlib_metadata import version
import pandas as pd
import sys
import src.utils.config as config
import time

def load_data(filepath, index_col=None):
    """Load an excel or a .csv file"""
    if filepath.endswith('.xlsx'):
        df = pd.read_excel(filepath)
    else:
        df = pd.read_csv(filepath, index_col=index_col)
    return df


def drop_nan(df, verbose=False):
    """Handles NaN in the dataset"""
    orig_size = len(df)
    df = df.dropna()  # only 24 rows contain NaN, for now I will drop them all
    if verbose:
        print('Dropped {} rows, {:.2f}% of original rows'. format(
            orig_size-len(df), (orig_size-len(df))/len(df)*100
        ))
    return df


def format_data(df):
    """Format the dataset (columns, index, etc.)"""
    # Format columns' names
    df.columns = df.columns.str.replace(' ', '_').str.lower() 
    df.columns = df.columns.str.replace('#_', '')
    # Change index if unique
    if df[config.index].is_unique:
        df = df.set_index(config.index)
    return df


def dump_value_counts(path, df, verbose=False):
    """Print value counts to a txt file for further exp"""
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    orig_stdout = sys.stdout
    with open(path+timestamp, 'w', encoding='utf-8') as f:
        sys.stdout = f
        for col in df.columns:
            print(col)
            print(df[col].value_counts(), end='\n'*3)
        sys.stdout = orig_stdout
    if verbose:
        print('Value counts information has been printed to {}'.format(
            path
        ))
    f.close()
