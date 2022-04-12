import pandas as pd

def print_df(df):
    """
        prints the complete dataframe df
    """
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)
