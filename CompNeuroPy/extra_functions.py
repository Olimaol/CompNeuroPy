import pandas as pd
from contextlib import contextmanager
import sys, os

def print_df(df):
    """
        prints the complete dataframe df
    """
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(df)
        
        
def flatten_list(lst):
    """
        lst: list of lists or mixed: values and lists
        retuns flattened list
    """
    
    ### if lists in lst --> upack them and retunr flatten_list of new list
    new_lst = []
    list_in_lst = False
    for val in lst:
        if isinstance(val,list):
            list_in_lst = True
            for sub_val in val:
                new_lst.append(sub_val)
        else:
            new_lst.append(val)
            
    if list_in_lst:
        return flatten_list(new_lst)
    ### else return lst
    else:
        return lst
    
    
def remove_key(d, key):
    """
        removes an element from a dict, returns the new dict
    """
    r = dict(d)
    del r[key]
    return r

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:  
            yield
        finally:
            sys.stdout = old_stdout
