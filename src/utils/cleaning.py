"""
Functions for loading and cleaning the dataset
"""

import pandas as pd
import sys
import config


def load_data(filepath):
    df = pd.read_excel(filepath)
    return df


def drop_nan(df):
    """Handles NaN in the dataset"""
    orig_size = len(df)
    df = df.dropna()  # only 24 rows contain NaN, for now I will drop them all
    # print('Dropped {} rows, {:.2f}% of original rows'. format(
    #     orig_size-len(df), (orig_size-len(df))/len(df)*100
    # ))
    return df


def format_data(df):
    """Format the dataset (columns, index, etc.)"""
    # Format columns' names
    df.columns = df.columns.str.replace(
        ' ', '_').str.lower()  # replace '#' with 'n'?
    # Change index if unique
    if df[config.index].is_unique:
        df = df.set_index(config.index)

    return df


def dump_value_counts(path, df):
    """Print value counts to a txt file for further exp"""
    orig_stdout = sys.stdout
    with open(path, 'w', encoding='utf-8') as f:
        sys.stdout = f
        for col in df.columns:
            print(col)
            print(df[col].value_counts(), end='\n'*3)
        sys.stdout = orig_stdout
    f.close()
