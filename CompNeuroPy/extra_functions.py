import pandas as pd
from contextlib import contextmanager
import sys
import os
from CompNeuroPy import analysis_functions as af
from ANNarchy import dt


def print_df(df):
    """
    prints the complete dataframe df
    """
    with pd.option_context(
        "display.max_rows", None, "display.max_columns", None
    ):  # more options can be specified also
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
        if isinstance(val, list):
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


def sci(nr):
    if af.get_number_of_zero_decimals(nr) == 0:
        return str(round(nr, 1))
    else:
        return f"{nr*10**af.get_number_of_zero_decimals(nr):.1f}e-{af.get_number_of_zero_decimals(nr)}"


def unpack_monDict_keys(s):
    """
    s : a key of a monDict
    """
    splitted_s = s.split(";")
    compartment_type = splitted_s[0]
    if not (compartment_type in ["pop", "proj"]):
        print(
            f"wrong compartment type in {compartment_type}\nhas to be 'pop' or 'proj'"
        )
        quit()
    compartment_name = splitted_s[1]
    if len(splitted_s) == 3:
        period = float(splitted_s[2])
    else:
        period = {"pop": dt(), "proj": dt() * 1000}[compartment_type]
    return compartment_type, compartment_name, period
