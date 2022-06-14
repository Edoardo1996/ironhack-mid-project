"""
This script cleans and formats the raw dataset.
"""

import pandas as pd
import sys


def load_data(filepath):
    df = pd.read_excel(filepath)
    return df


def drop_nan(df):
    """Handles NaN in the dataset"""
    orig_size = len(df)
    df = df.dropna()  # only 24 rows contain NaN, for now I will drop them all
    print('Dropped {} rows, {:.2f}% of original rows'. format(
        orig_size-len(df), (orig_size-len(df))/len(df)*100
    ))
    return df


def format_data(df):
    """Format the dataset (columns, index, etc.)"""
    # Format columns' names
    df.columns = df.columns.str.replace(
        ' ', '_').str.lower()  # replace '#' with 'n'?
    # Change index if unique
    if df['customer_number'].is_unique:
        df = df.set_index('customer_number')

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


def main():
    path = r'src/data/input/'
    filename = 'creditcardmarketing.xlsx'
    df = load_data(path+filename)
    df = drop_nan(df)
    df = format_data(df)
    dump_value_counts('src/data/tmp/value_counts.txt', df)
    # Save to .csv file for further elaboration
    df.to_csv('src/data/output/credictcardmarketing.csv')


if __name__ == '__main__':
    main()
