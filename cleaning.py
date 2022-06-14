"""
This script cleans and formats the raw dataset.
"""

import pandas as pd
import numpy as np

def load_data(filepath):
    df = pd.read_excel(filepath)
    return df

def format_data(df):
    """Format the dataset (columns, index, etc.)"""
    # Format columns' names
    df.columns = df.columns.str.replace(' ', '_').str.lower() # replace '#' with 'n'?
    # Change index if unique
    if df['customer_number'].is_unique:
        df = df.set_index('customer_number')
    pass
    
    return df
    


def main():
    path = r'src/data/input/'
    filename = 'creditcardmarketing.xlsx'
    df = load_data(path+filename)
    format_data(df)

if __name__ == '__main__':
    main()