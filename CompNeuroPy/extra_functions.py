import pandas as pd
from contextlib import contextmanager
import sys
import os
from CompNeuroPy import analysis_functions as af
from ANNarchy import dt
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, to_rgba_array
import numpy as np
from collections.abc import Sized


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
    s : a key of a monDict format:
        "compartment_type;compartment_name;period" or "compartment_type;compartment_name"

    return compartment_type, compartment_name, period

    if period not provided --> for pop return dt() for proj return dt()*1000
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

    period = int(period / dt()) * dt()
    return compartment_type, compartment_name, period


class Cmap:
    def __init__(self, cmap_name, vmin, vmax):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)

    def __call__(self, x, alpha=1):
        vals = self.get_rgb(x)
        if isinstance(vals, tuple):
            vals = vals[:3] + (alpha,)
        else:
            vals[:, -1] = alpha
        return vals


class data_obj(object):
    def __init__(self) -> None:
        pass

    def __setattr__(self, name: str, value) -> None:
        super().__setattr__(name, value)

    def __getattribute__(self, __name: str):
        try:
            return super().__getattribute__(__name)
        except:
            self.__setattr__(__name, data_obj())
            return super().__getattribute__(__name)


def create_cm(colors, name="my_cmap", N=256, gamma=1.0):
    """
    Create a `LinearSegmentedColormap` from a list of colors.

    Parameters
    ----------
    name : str
        The name of the colormap.
    colors : array-like of colors or array-like of (value, color)
        If only colors are given, they are equidistantly mapped from the
        range :math:`[0, 1]`; i.e. 0 maps to ``colors[0]`` and 1 maps to
        ``colors[-1]``.
        If (value, color) pairs are given, the mapping is from *value*
        to *color*. This can be used to divide the range unevenly.
    N : int
        The number of rgb quantization levels.
    gamma : float
    """
    if not np.iterable(colors):
        raise ValueError("colors must be iterable")

    if (
        isinstance(colors[0], Sized)
        and len(colors[0]) == 2
        and not isinstance(colors[0], str)
    ):
        # List of value, color pairs
        vals, colors = zip(*colors)
    else:
        vals = np.linspace(0, 1, len(colors))

    r_g_b_a = np.zeros((len(colors), 4))
    for color_idx, color in enumerate(colors):
        if isinstance(color, str):
            ### color given by name
            r_g_b_a[color_idx] = to_rgba_array(color)
        else:
            ### color given by rgb value
            color = np.array(color)
            if color.max() > 1:
                ### assume that max value is 255
                color = color / 255
            r_g_b_a[color_idx] = np.concatenate([color, np.array([1])])
    r = r_g_b_a[:, 0]
    g = r_g_b_a[:, 1]
    b = r_g_b_a[:, 2]
    a = r_g_b_a[:, 3]

    cdict = {
        "red": np.column_stack([vals, r, r]),
        "green": np.column_stack([vals, g, g]),
        "blue": np.column_stack([vals, b, b]),
        "alpha": np.column_stack([vals, a, a]),
    }

    return LinearSegmentedColormap(name, cdict, N, gamma)
