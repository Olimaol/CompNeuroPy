import pandas as pd

def print_df(df):
    """
        prints the complete dataframe df
    """
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)
        
        
def flatten_list(lst):
    """
        lst: list of lists
        retuns flattened list
    """
    return [item for sublist in lst for item in sublist]
    
    
def remove_key(d, key):
    """
        removes an element from a dict, returns the new dict
    """
    r = dict(d)
    del r[key]
    return r
