import pandas as pd
from contextlib import contextmanager
import sys
import os
from CompNeuroPy import analysis_functions as af
from CompNeuroPy import system_functions as sf
from CompNeuroPy import model_functions as mf
from CompNeuroPy.generate_model import CompNeuroModel
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, to_rgba_array
import numpy as np
from collections.abc import Sized
from typing import Callable
from tqdm import tqdm
from copy import deepcopy
from deap import base
from deap import creator
from deap import tools
from deap import cma
from ANNarchy import Neuron, Population, simulate, setup, get_population
from sympy import symbols, Symbol, solve, sympify, Eq, lambdify, factor
from scipy.interpolate import griddata
import re
from typingchecker import check_types
import warnings
import json
from matplotlib.widgets import Slider
from matplotlib.gridspec import GridSpec
from screeninfo import get_monitors
import cmaes
import efel


def print_df(df: pd.DataFrame | dict, **kwargs):
    """
    Prints the complete dataframe df

    Args:
        df (pandas dataframe or dict):
            Dataframe to be printed
    """
    if isinstance(df, dict):
        df = pd.DataFrame.from_dict(df)
    with pd.option_context(
        "display.max_rows", None
    ):  # more options can be specified also
        print(df, **kwargs)


def flatten_list(lst):
    """
    Retuns flattened list

    Args:
        lst (list of lists or mixed: values and lists):
            List to be flattened

    Returns:
        new_list (list):
            Flattened list
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
    Removes an element from a dict, returns the new dict

    Args:
        d (dict):
            Dict to be modified
        key (str):
            Key to be removed

    Returns:
        r (dict):
            Modified dict
    """
    r = dict(d)
    del r[key]
    return r


@contextmanager
def suppress_stdout():
    """
    Suppresses the print output of a function

    Examples:
        ```python
        with suppress_stdout():
            print("this will not be printed")
        ```
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def sci(nr):
    """
    Rounds a number to a single decimal.
    If number is smaller than 0 it is converted to scientific notation with 1 decimal.

    Args:
        nr (float or int):
            Number to be converted

    Returns:
        str (str):
            String of the number in scientific notation

    Examples:
        >>> sci(0.0001)
        '1.0e-4'
        >>> sci(1.77)
        '1.8'
        >>> sci(1.77e-5)
        '1.8e-5'
        >>> sci(177.22)
        '177.2'
    """
    if af.get_number_of_zero_decimals(nr) == 0:
        return str(round(nr, 1))
    else:
        return f"{nr*10**af.get_number_of_zero_decimals(nr):.1f}e-{af.get_number_of_zero_decimals(nr)}"


class Cmap:
    """
    Class to create a colormap with a given name and range. The colormap can be called
    with a value between 0 and 1 to get the corresponding rgb value.
    """

    def __init__(self, cmap_name, vmin, vmax):
        """
        Args:
            cmap_name (str):
                Name of the colormap
            vmin (float):
                Lower limit of the colormap
            vmax (float):
                Upper limit of the colormap
        """
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def __call__(self, x, alpha=1):
        """
        Returns the rgba value of the colormap at the given value.

        Args:
            x (float):
                Value between 0 and 1
            alpha (float):
                Alpha value of the rgba value

        Returns:
            rgba (tuple):
                RGBA value of the colormap at the given value
        """
        vals = self.get_rgb(x)
        if isinstance(vals, tuple):
            vals = vals[:3] + (alpha,)
        else:
            vals[:, -1] = alpha
        return vals

    def get_rgb(self, val):
        """
        Returns the rgb value of the colormap at the given value.

        Args:
            val (float):
                Value between 0 and 1

        Returns:
            rgb (tuple):
                RGB value of the colormap at the given value
        """
        return self.scalarMap.to_rgba(val)


class _DataCl(object):
    def __init__(self) -> None:
        pass

    def __setattr__(self, name: str, value) -> None:
        super().__setattr__(name, value)

    def __getattribute__(self, __name: str):
        try:
            return super().__getattribute__(__name)
        except:
            self.__setattr__(__name, _DataCl())
            return super().__getattribute__(__name)


### keep old name for compatibility
data_obj = _DataCl


def create_cm(colors, name="my_cmap", N=256, gamma=1.0, vmin=0, vmax=1):
    """
    Create a `LinearSegmentedColormap` from a list of colors.

    Args:
        colors (array-like of colors or array-like of (value, color)):
            If only colors are given, they are equidistantly mapped from the
            range :math:`[0, 1]`; i.e. 0 maps to ``colors[0]`` and 1 maps to
            ``colors[-1]``.
            If (value, color) pairs are given, the mapping is from *value*
            to *color*. This can be used to divide the range unevenly.
        name (str, optional):
            The name of the colormap, by default 'my_cmap'.
        N (int, optional):
            The number of rgb quantization levels, by default 256.
        gamma (float, optional):
            Gamma correction value, by default 1.0.
        vmin (float, optional):
            The minimum value of the colormap, by default 0.
        vmax (float, optional):
            The maximum value of the colormap, by default 1.

    Returns:
        linear_colormap (_LinearColormapClass):
            The colormap object
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
        vals = np.array(vals).astype(float)
        colors = list(colors)
        ### insert values for 0 and 1 if not given
        ### they equal the colors of the borders of the given range
        if vals.min() != 0.0:
            colors = [colors[np.argmin(vals)]] + colors
            vals = np.insert(vals, 0, 0.0)
        if vals.max() != 1.0:
            colors = colors + [colors[np.argmax(vals)]]
            vals = np.insert(vals, len(vals), 1.0)
    else:
        vals = np.linspace(0, 1, len(colors))

    ### sort values and colors, they have to increase
    sort_idx = np.argsort(vals)
    vals = vals[sort_idx]
    colors = [colors[idx] for idx in sort_idx]

    r_g_b_a = np.zeros((len(colors), 4))
    for color_idx, color in enumerate(colors):
        if isinstance(color, str):
            ### color given by name
            r_g_b_a[color_idx] = to_rgba_array(color)
        else:
            ### color given by rgb(maybe a) value
            color = np.array(color).astype(float)
            ### check color size
            if len(color) != 3 and len(color) != 4:
                raise ValueError(
                    "colors must be names or consist of 3 (rgb) or 4 (rgba) numbers"
                )
            if color.max() > 1:
                ### assume that max value is 255
                color[:3] = color[:3] / 255
            if len(color) == 4:
                ### gamma already given
                r_g_b_a[color_idx] = color
            else:
                ### add gamma
                r_g_b_a[color_idx] = np.concatenate([color, np.array([gamma])])
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

    return _LinearColormapClass(name, cdict, N, gamma, vmin, vmax)


class _LinearColormapClass(LinearSegmentedColormap):
    def __init__(self, name, segmentdata, N=..., gamma=..., vmin=0, vmax=1) -> None:
        """
        Args:
            name (str):
                The name of the colormap.
            segmentdata (dict):
                Mapping from scalar values to colors.
                The scalar values are typically in the interval (0, 1),
                but other intervals are allowed.
                The colors may be specified in any way understandable by
                `matplotlib.colors.ColorConverter.to_rgba`.
            N (int, optional):
                The number of rgb quantization levels, by default ...
            gamma (float, optional):
                Gamma correction value, by default ...
            vmin (float, optional):
                The minimum value of the colormap, by default 0.
            vmax (float, optional):
                The maximum value of the colormap, by default 1.
        """
        self.my_vmin = vmin
        self.my_vmax = vmax
        super().__init__(name, segmentdata, N, gamma)

    def __call__(self, X, alpha=None, bytes=False):
        """
        Args:
            X (scalar, ndarray):
                The data value(s) to convert to RGBA.
                For floats, X should be in the interval ``[0.0, 1.0]`` to
                return the RGBA values ``X*100`` percent along the Colormap line.
                For integers, X should be in the interval ``[0, Colormap.N)`` to
                return RGBA values *indexed* from the Colormap with index ``X``.
            alpha (float, None):
                Alpha must be a scalar between 0 and 1, or None.
            bytes (bool):
                If False (default), the returned RGBA values will be floats in the
                interval ``[0, 1]`` otherwise they will be uint8s in the interval
                ``[0, 255]``.

        Returns:
            Tuple of RGBA values if X is scalar, otherwise an array of
            RGBA values with a shape of ``X.shape + (4, )``.
        """
        ### rescale X in the range [0,1]
        ### using vmin and vmax
        if self.my_vmin != 0 or self.my_vmax != 1:
            X = (X - self.my_vmin) / (self.my_vmax - self.my_vmin)
        return super().__call__(X, alpha, bytes)


### keep old name for compatibility
my_linear_cmap_obj = _LinearColormapClass


class DecisionTree:
    """
    Class to create a decision tree.
    """

    def __init__(self):
        """
        Create a new empty decision tree.
        """
        ### node list is a list of lists
        ### first idx = level of tree
        ### second idx = all nodes in the level
        self.node_list = [[]]

    def node(self, parent=None, prob=0, name=None):
        """
        Create a new node in the decision tree.

        Args:
            parent (node object):
                Parent node of the new node
            prob (float):
                Probability of the new node
            name (str):
                Name of the new node

        Returns:
            new_node (node object):
                The new node
        """

        ### create new node
        new_node = DecisionTreeNode(tree=self, parent=parent, prob=prob, name=name)
        ### add it to node_list
        if len(self.node_list) == new_node.level:
            self.node_list.append([])
        self.node_list[new_node.level].append(new_node)
        ### return the node object
        return new_node

    def get_path_prod(self, name):
        """
        Get the path and path product of a node with a given name.

        Args:
            name (str):
                Name of the node

        Returns:
            path (str):
                Path to the node
            path_prod (float):
                Path product of the node
        """

        ### search for all nodes with name
        ### start from behind
        search_node_list = []
        path_list = []
        path_prod_list = []
        for level in range(len(self.node_list) - 1, -1, -1):
            for node in self.node_list[level]:
                if node.name == name:
                    search_node_list.append(node)
        ### get the paths and path products for the found nodes
        for node in search_node_list:
            path, path_prod = self._get_path_prod_rec(node)
            path_list.append(path)
            path_prod_list.append(path_prod)
        ### return the paths and path products
        return [
            [path_list[idx], path_prod_list[idx]]
            for idx in range(len(search_node_list))
        ]

    def _get_path_prod_rec(self, node):
        """
        Recursive function to get the path and path product of a node.

        Args:
            node (node object):
                Node to get the path and path product of

        Returns:
            path_str (str):
                Path to the node
            prob (float):
                Path product of the node
        """
        node: DecisionTreeNode = node

        if node.parent == None:
            return ["/" + node.name, node.prob]
        else:
            path_str, prob = self._get_path_prod_rec(node.parent)
            return [path_str + "/" + node.name, prob * node.prob]


### keep old name for compatibility
decision_tree = DecisionTree


class DecisionTreeNode:
    """
    Class to create a node in a decision tree.
    """

    id_counter = 0

    def __init__(self, tree: DecisionTree, parent=None, prob=0, name=""):
        """
        Create a new node in a decision tree.

        Args:
            tree (DecisionTree object):
                Decision tree the node belongs to
            parent (node object):
                Parent node of the new node
            prob (float):
                Probability of the new node
            name (str):
                Name of the new node
        """
        self.tree = tree
        parent: DecisionTreeNode = parent
        self.parent = parent
        self.prob = prob
        self.name = name
        self.id = int(self.id_counter)
        self.id_counter += 1
        if parent != None:
            self.level = int(parent.level + 1)
        else:
            self.level = int(0)

    def add(self, name, prob):
        """
        Add a child node to the node.

        Args:
            name (str):
                Name of the new node
            prob (float):
                Probability of the new node

        Returns:
            new_node (node object):
                The new node
        """

        return self.tree.node(parent=self, prob=prob, name=name)

    def get_path_prod(self):
        """
        Get the path and path product of the node.

        Returns:
            path (str):
                Path to the node
            path_prod (float):
                Path product of the node
        """
        return self.tree._get_path_prod_rec(self)


### keep old name for compatibility
node_cl = DecisionTreeNode


def evaluate_expression_with_dict(expression, value_dict):
    """
    Evaluate a mathematical expression using values from a dictionary.

    This function takes a mathematical expression as a string and a dictionary
    containing variable names as keys and corresponding values as numpy arrays.
    It replaces the variable names in the expression with their corresponding
    values from the dictionary and evaluates the expression.

    Args:
        expression (str):
            A mathematical expression to be evaluated. Variable
            names in the expression should match the keys in the value_dict.
        value_dict (dict):
            A dictionary containing variable names (strings) as
            keys and corresponding numpy arrays or numbers as values.

    Returns:
        result (value or array):
            The result of evaluating the expression using the provided values.

    Examples:
        >>> my_dict = {"a": np.ones(10), "b": np.arange(10)}
        >>> my_string = "a*2-b+10"
        >>> evaluate_expression_with_dict(my_string, my_dict)
        array([12., 11., 10.,  9.,  8.,  7.,  6.,  5.,  4.,  3.])
    """
    # Replace dictionary keys in the expression with their corresponding values
    ### replace names with dict entries
    expression = _replace_names_with_dict(
        expression=expression, name_of_dict="value_dict", dictionary=value_dict
    )

    ### evaluate the new expression
    try:
        result = eval(expression)
        return result
    except Exception as e:
        raise ValueError(f"Error while evaluating expression: {str(e)}")


def _replace_names_with_dict(expression, name_of_dict, dictionary):
    """
    Args:
        expression (str):
            String which contains an equation using keys from the dict
        name_of_dict (str):
            Name of the dictionary
        dictionary (dict):
            The dictionary containing the keys used in the equation

    Returns:
        new_expression (str):
            Same as expression but the keys are replaced by name_of_dict['key']

    Examples:
        >>> _replace_names_with_dict(expression="a+b", name_of_dict="my_dict", dictionary={"a":5,"b":7})
        "my_dict['a']+my_dict['b']"
    """
    new_expression = expression

    new_expression = new_expression.replace("np.clip", "{np.clip}")
    new_expression = new_expression.replace("np.max", "{np.max}")
    new_expression = new_expression.replace("np.min", "{np.min}")
    new_expression = new_expression.replace("None", "{None}")

    sorted_names_list = sorted(list(dictionary.keys()), key=len, reverse=True)
    ### first replace largest variable names
    ### --> if smaller variable names are within larger variable names this should not cause a problem
    for name in sorted_names_list:
        if name in new_expression:
            ### replace the name in the new_equation_str
            ### only replace things which are not between {}
            new_expression = _replace_substrings_except_within_braces(
                new_expression, {name: f"{{{name_of_dict}['{name}']}}"}
            )
    ### remove curly braces again
    new_expression = new_expression.replace("{", "")
    new_expression = new_expression.replace("}", "")
    return new_expression


def _replace_substrings_except_within_braces(input_string, replacement_mapping):
    """
    Replace substrings in a string with other substrings, but only if the
    substring is not within braces.

    Args:
        input_string (str):
            The string in which substrings should be replaced.
        replacement_mapping (dict):
            A dictionary mapping substrings to be replaced to their replacements.

    Returns:
        result (str):
            The input string with substrings replaced.

    Examples:
        >>> _replace_substrings_except_within_braces("a+b", {"a":"c"})
        "c+b"
        >>> _replace_substrings_except_within_braces("a+b", {"a":"c", "b":"d"})
        "c+d"
        >>> _replace_substrings_except_within_braces("a+{b}", {"a":"c", "b":"d"})
        "c+{b}"
    """

    result = []
    inside_braces = False
    i = 0

    while i < len(input_string):
        if input_string[i] == "{":
            inside_braces = True
            result.append(input_string[i])
            i += 1
        elif input_string[i] == "}":
            inside_braces = False
            result.append(input_string[i])
            i += 1
        else:
            if not inside_braces:
                found_match = False
                for old_substr, new_substr in replacement_mapping.items():
                    if input_string[i : i + len(old_substr)] == old_substr:
                        result.append(new_substr)
                        i += len(old_substr)
                        found_match = True
                        break
                if not found_match:
                    result.append(input_string[i])
                    i += 1
            else:
                result.append(input_string[i])
                i += 1

    return "".join(result)


class DeapCma:
    """
    Class to run the deap Covariance Matrix Adaptation Evolution Strategy optimization.

    Attributes:
        deap_dict (dict):
            Dictionary containing the toolbox, the hall of fame, the statistics, the
            lower and upper bounds, the parameter names, the inverse scaler and the
            strategy.
    """

    def __init__(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        evaluate_function: Callable,
        max_evals: None | int = None,
        p0: None | np.ndarray = None,
        param_names: None | list[str] = None,
        learn_rate_factor: float = 1,
        damping_factor: float = 1,
        verbose: bool = False,
        plot_file: None | str = "logbook.png",
        cma_params_dict: dict = {},
        source_solutions: list[tuple[np.ndarray, float]] = [],
    ):
        """

        Args:
            lower (np.ndarray):
                Lower bounds of the parameters
            upper (np.ndarray):
                Upper bounds of the parameters
            evaluate_function (Callable):
                Function evaluating the losses of a population of individuals. Return value
                should be a list of tuples with the losses of the individuals.
            max_evals (int, optional):
                Maximum number of evaluations. If not given here, it has to be given in
                the run function. By default None.
            p0 (None | np.ndarray, optional):
                Initial guess for the parameters. By default the mean of lower and upper
                bounds.
            param_names (None | list[str], optional):
                Names of the parameters. By default None.
            learn_rate_factor (float, optional):
                Learning rate factor (decrease -> slower). By default 1.
            damping_factor (float, optional):
                Damping factor (increase -> slower). By default 1.
            verbose (bool, optional):
                Whether or not to print details. By default False.
            plot_file (None | str, optional):
                File to save the deap plot to. If not given here, it has to be given in
                the run function. By default "logbook.png".
            cma_params_dict (dict, optional):
                Parameters for the deap cma strategy (deap.cma.Strategy). See [here](https://deap.readthedocs.io/en/master/api/algo.html#deap.cma.Strategy) for more
                details
            source_solutions (list[tuple[np.ndarray, float]], optional):
                List of tuples with the parameters and losses of source solutions. These
                solutions are used to initialize the covariance matrix. By default [].
        """
        ### store attributes
        self.max_evals = max_evals
        self.lower = lower
        self.upper = upper
        self.evaluate_function = evaluate_function
        self.p0 = p0
        self.param_names = param_names
        self.learn_rate_factor = learn_rate_factor
        self.damping_factor = damping_factor
        self.verbose = verbose
        self.plot_file = plot_file
        self.cma_params_dict = cma_params_dict
        self.source_solutions = source_solutions

        ### prepare the optimization
        self.deap_dict = self._prepare()

    def _prepare(self):
        """
        Prepares the deap Covariance Matrix Adaptation Evolution Strategy optimization.

        Returns:
            dict:
                Dictionary containing the toolbox, the hall of fame, the statistics, the
                lower and upper bounds, the parameter names, the inverse scaler and the
                strategy.
        """

        ### get attributes
        lower = self.lower
        upper = self.upper
        evaluate_function = self.evaluate_function
        p0 = self.p0
        param_names = self.param_names
        learn_rate_factor = self.learn_rate_factor
        damping_factor = self.damping_factor
        verbose = self.verbose
        cma_params_dict = self.cma_params_dict

        ### create scaler to scale parameters into range [0,1] based on lower and upper bounds
        upper_orig = deepcopy(upper)
        lower_orig = deepcopy(lower)

        def scaler(x):
            return (x - lower_orig) / (upper_orig - lower_orig)

        ### create inverse scaler to scale parameters back into original range [lower,upper]
        def inv_scaler(x):
            return x * (upper_orig - lower_orig) + lower_orig

        ### scale upper and lower bounds
        lower = scaler(lower)
        upper = scaler(upper)

        ### create the individual class, since this is eventually called multiple times
        ### deactivate warnings (it warns that the classes already exist)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
            creator.create("Individual", list, fitness=creator.FitnessMin)

        ### create the toolbox
        toolbox = base.Toolbox()
        ### function calculating losses from individuals (from whole population)
        toolbox.register("evaluate", evaluate_function)
        ### search strategy
        ### warm start with initial source solutions
        if len(self.source_solutions) > 0:
            ### scale source solutions
            for source_solution_idx in range(len(self.source_solutions)):
                self.source_solutions[source_solution_idx] = (
                    scaler(self.source_solutions[source_solution_idx][0]),
                    self.source_solutions[source_solution_idx][1],
                )
            centroid, sigma, cmatrix = cmaes.get_warm_start_mgd(
                source_solutions=self.source_solutions,
                gamma=1,
            )
            cma_params_dict["cmatrix"] = cmatrix
        else:
            centroid = (lower + upper) / 2 if isinstance(p0, type(None)) else scaler(p0)
            sigma = (upper - lower) / 4

        ### create the strategy
        strategy = cma.Strategy(
            centroid=centroid,
            sigma=sigma,
            **cma_params_dict,
        )

        ### slow down the learning rate and increase the damping
        strategy.ccov1 *= learn_rate_factor
        strategy.ccovmu *= learn_rate_factor
        strategy.damps *= damping_factor  # TODO what slows down?
        if verbose:
            print(
                f"lambda (The number of children to produce at each generation): {strategy.lambda_}"
            )
            print(
                f"mu (The number of parents to keep from the lambda children): {strategy.mu}"
            )
            print(f"weights: {strategy.weights}")
            print(f"mueff: {strategy.mueff}")
            print(f"ccum (Cumulation constant for covariance matrix.): {strategy.cc}")
            print(f"cs (Cumulation constant for step-size): {strategy.cs}")
            print(f"ccov1 (Learning rate for rank-one update): {strategy.ccov1}")
            print(f"ccovmu (Learning rate for rank-mu update): {strategy.ccovmu}")
            print(f"damps (Damping for step-size): {strategy.damps}")
        ### function generating a population during optimization
        toolbox.register("generate", strategy.generate, creator.Individual)
        ### function updating the search strategy
        toolbox.register("update", strategy.update)
        ### hall of fame to track best individual i.e. parameters
        hof = tools.HallOfFame(1)
        ### statistics to track evolution of loss
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        return {
            "toolbox": toolbox,
            "hof": hof,
            "stats": stats,
            "lower": lower,
            "upper": upper,
            "param_names": param_names,
            "inv_scaler": inv_scaler,
            "strategy": strategy,
        }

    def run(
        self,
        max_evals: None | int = None,
        verbose: None | bool = None,
        plot_file: None | str = None,
    ):
        """
        Runs the optimization with deap.

        Args:
            max_evals (int):
                Number of runs (here generations) a single optimization performs. By
                default None, i.e. the value from the initialization is used.
            verbose (bool, optional):
                Whether or not to print details. By default None, i.e. the value from
                the initialization is used.
            plot_file (str):
                Path to save the logbook plot to. By default None, i.e. the value from
                the initialization is used.

        Returns:
            best (dict):
                Dictionary containing the best parameters, the logbook, the last population
                of individuals and the best fitness.
        """

        ### get attributes
        max_evals = self.max_evals if max_evals is None else max_evals
        verbose = self.verbose if verbose is None else verbose
        plot_file = self.plot_file if plot_file is None else plot_file
        deap_dict = self.deap_dict

        ### run the search algorithm with the prepared deap_dict
        pop, logbook = self._deap_ea_generate_update(
            deap_dict,
            ngen=max_evals,
            verbose=verbose,
        )

        ### scale parameters of hall of fame back into original range [lower,upper]
        hof_final = deap_dict["inv_scaler"](deap_dict["hof"][0])
        best_fitness = deap_dict["hof"][0].fitness.values[0]

        ### get best parameters, last population of inidividuals and logbook
        best = {}
        for param_idx in range(len(deap_dict["lower"])):
            if deap_dict["param_names"] is not None:
                param_key = deap_dict["param_names"][param_idx]
            else:
                param_key = f"param{param_idx}"
            best[param_key] = hof_final[param_idx]
        best["logbook"] = logbook
        best["deap_pop"] = pop
        best["best_fitness"] = best_fitness

        ### plot logbook with logaritmic y-axis
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_yscale("log")
        ax.plot(logbook.select("gen"), logbook.select("min"), label="min")
        ax.plot(logbook.select("gen"), logbook.select("avg"), label="avg")
        ax.plot(logbook.select("gen"), logbook.select("max"), label="max")
        ax.legend()
        ax.set_xlabel("Generation")
        ax.set_ylabel("Loss")
        fig.tight_layout()
        sf.create_dir("/".join(plot_file.split("/")[:-1]))
        fig.savefig(plot_file, dpi=300)

        return best

    def _deap_ea_generate_update(
        self,
        deap_dict: dict,
        ngen: int,
        verbose: bool = False,
    ):
        """
        This function is copied from deap.algorithms.eaGenerateUpdate and modified.
        This is algorithm implements the ask-tell model proposed in
        [Colette2010]_, where ask is called `generate` and tell is called `update`.

        .. [Colette2010] Collette, Y., N. Hansen, G. Pujol, D. Salazar Aponte and
        R. Le Riche (2010). On Object-Oriented Programming of Optimizers -
        Examples in Scilab. In P. Breitkopf and R. F. Coelho, eds.:
        Multidisciplinary Design Optimization in Computational Mechanics,
        Wiley, pp. 527-565;

        Args:
            deap_dict (dict):
                Dictionary containing the deap toolbox, hall of fame, statistics, lower
                and upper bounds, parameter names, inverse scaler and strategy.
            ngen (int):
                number of runs (here generations) a single optimization performs
            verbose (bool, optional):
                Whether or not to print details. By default False.

        Returns:
            population:
                A list of individuals.
            logbook:
                A Logbook() object that contains the evolution statistics.
        """

        ### get variables from deap_dict
        toolbox = deap_dict["toolbox"]
        lower = deap_dict["lower"]
        upper = deap_dict["upper"]
        inv_scaler = deap_dict["inv_scaler"]
        stats = deap_dict["stats"]
        halloffame = deap_dict["hof"]
        strategy = deap_dict["strategy"]

        ### init logbook
        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

        ### define progress bar
        progress_bar = tqdm(range(ngen), total=ngen, unit="gen")
        early_stop = False

        ### loop over generations
        for gen in progress_bar:
            ### Generate a new population
            population = toolbox.generate()
            ### clip individuals of population to variable bounds
            ### TODO only if bounds are hard
            for ind in population:
                for idx, val in enumerate(ind):
                    ind[idx] = np.clip(val, lower[idx], upper[idx])
            ### Evaluate the individuals (here whole population at once)
            ### scale parameters back into original range [lower,upper]
            population_inv_scaled = [inv_scaler(ind) for ind in deepcopy(population)]
            fitnesses = toolbox.evaluate(population_inv_scaled)

            ### set fitnesses of individuals
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = fit

            ### check if nan in population
            for ind in population:
                nan_in_pop = np.isnan(ind.fitness.values[0])

            ### Update the hall of fame with the generated individuals
            if halloffame is not None and not nan_in_pop:
                halloffame.update(population)

            ### Update the strategy with the evaluated individuals
            toolbox.update(population)

            ### Stop if diagD is too small
            if np.min(strategy.diagD) < 1e-5:
                early_stop = True
                break

            ### Append the current generation statistics to the logbook
            record = stats.compile(population) if stats is not None else {}
            logbook.record(gen=gen, nevals=len(population), **record)
            if verbose:
                print(logbook.stream)

            ### update progress bar with current best loss
            progress_bar.set_postfix_str(
                f"best loss: {halloffame[0].fitness.values[0]:.5f}"
            )
        if early_stop and verbose:
            print("Stopping because convergence is reached.")

        return population, logbook


class VClampParamSearch:
    """
    Class to obtain the parameters of some neuron model equations (describing the change
    of the membrane potential v) by simulating voltage steps with a given neuron_model.
    An voltage clamp version of the equations is used to calculate instantaneous and
    holding "currents" for specific voltage steps. The parameters are then optimized
    to fit the calculated "currents" to the measured currents from the simulated neuron
    model.

    Attributes:
        p_opt (dict):
            The optimized parameters
    """

    @check_types()
    def __init__(
        self,
        neuron_model: Neuron,
        equations: str = """
        C*dv/dt = k*(v - v_r)*(v - v_t) - u
        du/dt = a*(b*(v - v_r) - u)
        """,
        bounds: dict[str, tuple[float, float]] = {
            "C": (0.1, 100),
            "v_r": (-90, -40),
            "v_t": (-90, -40),
            "k": (0.01, 1),
            "a": (0.01, 1),
            "b": (-5, 5),
        },
        p0: None | dict[str, float | list] = None,
        max_evals: int = 100,
        m: int = 20,
        n: int = 20,
        do_plot: bool = False,
        results_file: str = "v_clamp_search_results",
        plot_file: str = "v_clamp_search_plot.png",
        cma_params_dict: dict = {"learn_rate_factor": 1, "damping_factor": 1},
        compile_folder_name: str = "VClampParamSearch",
        verbose: bool = False,
    ):
        """
        Args:
            neuron_model (Neuron):
                The neuron model which is simulated to obtain the parameters for the
                equations
            equations (str, optional):
                The equations whose parameters should be obtained. Default: Izhikevich
                2007 neuron model
            bounds (dict, optional):
                The bounds for the parameters. For each parameter a bound should be
                given! Default: Izhikevich 2007 neuron model
            p0 (dict, optional):
                The initial guess for the parameters. Dict keys should be the same as
                the keys of bounds. The values can be either a single number for each
                parameter or a list of numbers. If lists are given, all have to have
                the same length, which will be the number of initial guesses for the
                parameters, i.e. how often the optimization is run. Default: None,
                i.e. the mid of the bounds is used as a single initial guess.
            max_evals (int, optional):
                The maximum number of evaluations for a single optimization run.
                Default: 100
            m (int, optional):
                The number of initial voltages for the voltage step simulations.
                Default: 20
            n (int, optional):
                The number of voltage steps for the voltage step simulations.
                Defaults: 20
            do_plot (bool, optional):
                If True, plots are created. Default: False
            results_file (str, optional):
                The name of the file where the results are stored, without file ending.
                Default: "v_clamp_search_results"
            plot_file (str, optional):
                The name of the file where the plot is stored, with file ending.
                Default: "v_clamp_search_plot.png"
            cma_params_dict (dict, optional):
                Parameters for the deap cma strategy (deap.cma.Strategy). See [here](https://deap.readthedocs.io/en/master/api/algo.html#deap.cma.Strategy)
                for more details. Additional parameters are learn_rate_factor and
                damping_factor. Default: {"learn_rate_factor": 1, "damping_factor": 1}
            compile_folder_name (str, optional):
                The name of the folder within "annarchy_folders" where the ANNarchy
                network is compiled to. Default: "VClampParamSearch"
            verbose (bool, optional):
                If True, print details. Default: False
        """
        self.verbose = verbose
        self._verbose_extreme = False
        ### store the given neuron model and a voltage clamp version of it
        self.neuron_model = neuron_model
        self._neuron_model = deepcopy(neuron_model)
        self._neuron_model_clamp = self._get_neuron_model_clamp()

        ### store other attributes
        self.m = m
        self.n = n
        self.equations = equations
        self.p0 = p0
        ### check if p0 is correct and if lists are given, create also lists single
        ### numbers which are given
        self._p0 = self._get_p0()
        self.max_evals = max_evals
        self.bounds = bounds
        self.do_plot = do_plot
        self.results_file = results_file
        self.plot_file = plot_file
        self.cma_params_dict = cma_params_dict
        ### check if file names are correct
        if "." in self.results_file or "." not in self.plot_file:
            raise ValueError(
                "results_file should not contain file ending and plot_file should!"
            )
        self.compile_folder_name = compile_folder_name
        self._timestep = 0.001

        ### create folder for plots
        if self.do_plot:
            sf.create_dir("/".join(plot_file.split("/")[:-1]))

        ### create the functions for v_clamp_inst and v_clamp_hold using the given
        ### izhikevich equations
        self._f_inst, self._f_hold, self._f_variables = self._create_v_clamp_functions()

        ### create the voltage step arrays
        self._v_0_arr, self._v_step_arr = self._create_voltage_step_arrays()

        ### for each neuron model create a population
        if self.verbose:
            print("Creating models...")
        mf.cnp_clear()
        self._model_normal, self._model_clamp = self._create_model()

        ### perform resting state and voltage step simulations to obtain v_clamp_inst,
        ### v_clamp_hold and v_rest
        self._v_clamp_inst_arr = None
        self._v_clamp_hold_arr = None
        if self.verbose:
            print("Performing simulations...")
        (
            self._v_rest,
            self._v_clamp_inst_arr,
            self._v_clamp_hold_arr,
            self._v_step_unique,
            self._v_clamp_hold_unique,
        ) = self._simulations()

        ### tune the free paramters of the functions for v_clamp_inst and v_clamp_hold
        ### to fit the data
        if self.verbose:
            print("Tuning parameters...")
        self._p_opt = self._tune_v_clamp_functions()
        self.p_opt = {
            param_name: self._p_opt.get(param_name, None)
            for param_name in self.bounds.keys()
        }
        self.p_opt["best_fitness"] = self._p_opt["best_fitness"]

        ### print and save optimized parameters
        if self.verbose:
            print(f"Optimized parameters: {self.p_opt}")
        ### save as pkl file
        sf.save_variables(
            [self.p_opt],
            [results_file.split("/")[-1]],
            "/".join(results_file.split("/")[:-1]) if "/" in results_file else "./",
        )
        ### save human readable as json file
        json.dump(
            self.p_opt,
            open(
                f"{results_file}.json",
                "w",
            ),
            indent=4,
        )

        ### create a neuron model with the tuned parameters and the given equations
        ### then run the simulations again with this neuron model
        if self.verbose:
            print("Running simulations with tuned parameters...")
        mf.cnp_clear()
        self._neuron_model = self._create_neuron_model_with_tuned_parameters()
        self._neuron_model_clamp = self._get_neuron_model_clamp()
        self._model_normal, self._model_clamp = self._create_model()
        self._simulations()

    def _get_p0(self):
        """
        Check if p0 is correct and if lists are given, create also lists single numbers
        which are given.

        Returns:
            _p0 (dict):
                The corrected p0
        """
        _p0 = None
        if self.p0 is not None:
            ### collect lengths of lists
            list_lengths = []
            for key, val in self.p0.items():
                if isinstance(val, list):
                    list_lengths.append(len(val))
            ### check if all lists have the same length
            if len(set(list_lengths)) > 1:
                raise ValueError("All lists in p0 should have the same length!")
            ### create new p0 with lists for all parameters
            _p0 = deepcopy(self.p0)
            for key, val in _p0.items():
                if not isinstance(val, list):
                    _p0[key] = [val] * list_lengths[0] if list_lengths else [val]
        return _p0

    def _create_neuron_model_with_tuned_parameters(self):
        """
        Create a neuron model with the tuned parameters and the given equations.

        Returns:
            neuron_mondel (Neuron):
                the neuron model with the tuned parameters and the given equations
        """
        ### create the neuron with the tuned parameters, if a parameter is not tuned
        ### use the mid of the bounds (these parameters should not affect v_clamp_inst
        ### and v_clamp_hold)
        parameters = "\n".join(
            [
                f"{key} = {self._p_opt.get(key,sum(self.bounds[key])/2)}"
                for key in self.bounds.keys()
            ]
        )
        neuron_mondel = Neuron(
            parameters=parameters,
            equations=self.equations + "\nr=0",
        )

        return neuron_mondel

    def _tune_v_clamp_functions(self):
        """
        Tune the free paramters of the functions for v_clamp_inst and v_clamp_hold
        to fit the data.
        """
        ### get the names of the free parameters which will be tuned
        sub_var_names_list = []
        for var in self._f_variables:
            if str(var) not in self.bounds or str(var) == "v_r":
                continue
            sub_var_names_list.append(str(var))

        ### target array for the error function below
        target_arr = np.concatenate([self._v_clamp_inst_arr, self._v_clamp_hold_unique])

        ### create a function for the error
        def error_function(x):
            if self._verbose_extreme:
                print(f"Current guess: {x}")
            ### set the free parameters of the functions
            p_dict = {
                var_name: x[var_idx]
                for var_idx, var_name in enumerate(sub_var_names_list)
            }
            if self._verbose_extreme:
                print(f"Current guess dict: {p_dict}")
            var_dict = {str(var): p_dict.get(str(var)) for var in self._f_variables}
            var_dict["v_r"] = self._v_rest
            if self._verbose_extreme:
                print(f"var_dict: {var_dict}")
                print(f"f_variables: {self._f_variables}")

            ### calculate the voltage clamp values
            ### 1st f_inst, it depends on v_0 and v_step
            var_dict["v_0"] = self._v_0_arr
            var_dict["v_step"] = self._v_step_arr
            f_inst_arr = self._f_inst(*list(var_dict.values()))
            ### 2nd f_hold, it depends only on v_step
            var_dict["v_0"] = self._v_0_arr[int(len(self._v_0_arr) / 2)]
            var_dict["v_step"] = self._v_step_unique
            f_hold_arr = self._f_hold(*list(var_dict.values()))

            ### calculate the error
            error = af.rmse(target_arr, np.concatenate([f_inst_arr, f_hold_arr]))
            return error

        def error_function_deap(population):
            error_list = [(error_function(individual),) for individual in population]
            return error_list

        ### perform the optimization
        ### set bounds
        bounds = np.array([self.bounds[var_name] for var_name in sub_var_names_list])
        ### set initial guess
        if isinstance(self._p0, type(None)):
            ### if no initial guess is given use the middle of the bounds
            initial_guess = np.array(
                [sum(self.bounds[var_name]) / 2.0 for var_name in sub_var_names_list]
            )
        else:
            ### initial guess is an array 1st dimension is the number of tuned parameters
            ### 2nd dimension is the number of initial guesses
            initial_guess = np.array(
                [self._p0[var_name] for var_name in sub_var_names_list]
            )
        if self.verbose:
            print(f"p0: {self.p0}")
            print(f"_p0: {self._p0}")
            print(f"bounds: {self.bounds}")
            print(f"Initial guess: {initial_guess}")
            print(f"Bounds: {bounds}\n")

        ### run the optimization multiple times with different initial guesses
        print_results = []
        best_fitness = np.inf
        for initial_guess_idx in range(initial_guess.shape[1]):
            deap_cma = DeapCma(
                max_evals=self.max_evals,
                lower=bounds[:, 0],
                upper=bounds[:, 1],
                evaluate_function=error_function_deap,
                p0=initial_guess[:, initial_guess_idx],
                param_names=sub_var_names_list,
                learn_rate_factor=self.cma_params_dict["learn_rate_factor"],
                damping_factor=self.cma_params_dict["damping_factor"],
                verbose=False,
                plot_file=self.plot_file.split(".")[0]
                + f"_logbook_{initial_guess_idx}."
                + self.plot_file.split(".")[-1],
                cma_params_dict=self.cma_params_dict,
            )
            result = deap_cma.run()
            print_results_dict = {
                var_name: result[var_name] for var_name in sub_var_names_list
            }
            print_results_dict["best_fitness"] = result["best_fitness"]
            print_results.append(print_results_dict)
            if result["best_fitness"] < best_fitness:
                best_fitness = result["best_fitness"]
                best_result = result
        result_dict = {
            var_name: best_result[var_name] for var_name in sub_var_names_list
        }
        result_dict["best_fitness"] = best_result["best_fitness"]
        result_dict["v_r"] = self._v_rest

        if self.verbose:
            print("Results:")
            print_df(pd.DataFrame(print_results))
            print(f"Result: {result_dict}")

        return result_dict

    def _create_v_clamp_functions(self):
        """
        Create the functions for v_clamp_inst and v_clamp_hold using the given
        izhikevich equations.

        Returns:
            f_inst (Callable):
                Function for v_clamp_inst
            f_hold (Callable):
                Function for v_clamp_hold
            variables (list):
                List of variables used for the functions
        """
        ### obtain all variables and parameters from the equation string
        variables_name_list = self._get_variables_from_eq(self.equations)

        ### split equations into lines, remove whitespace and only keep entries with
        ### length > 0
        eq_line_list = self.equations.splitlines()
        eq_line_list = [line.replace(" ", "") for line in eq_line_list]
        eq_line_list = [line for line in eq_line_list if len(line) > 0]

        ### create a dictionary with the variables as keys and the sympy symbols as
        ### values
        variables_sympy_dict = {key: Symbol(key) for key in variables_name_list}

        ### also create sympy symbols for v_clamp, v_0 and v_step
        variables_sympy_dict["v_clamp"] = Symbol("v_clamp")
        variables_sympy_dict["v_0"] = Symbol("v_0")
        variables_sympy_dict["v_step"] = Symbol("v_step")

        ### sympify equations
        eq_sympy_list = []
        variables_to_solve_for_list = []
        instant_update_list = []
        for line_idx, line in enumerate(eq_line_list):
            left_side = line.split("=")[0]
            right_side = line.split("=")[1]
            ### check if line contains dv/dt, replace it with v_clamp and add v_clamp
            ### to variables_to_solve_for_list, also set instant_update to True
            if "dv/dt" in line:
                variables_to_solve_for_list.append("v_clamp")
                left_side = left_side.replace("dv/dt", "v_clamp")
                instant_update_list.append(True)
            ### check if line contains any other derivative with syntax "d<var>/dt"
            ### using re, replace it with 0 and add the variable to
            ### variables_to_solve_for_list, also set instant_update to False
            elif re.search(r"d\w+/dt", line):
                variables_to_solve_for_list.append(
                    re.search(r"d(\w+)/dt", line).group(1)
                )
                left_side = left_side.replace(
                    re.search(r"d(\w+)/dt", line).group(0), "0"
                )
                instant_update_list.append(False)
            ### else it is a "normal" equation (<var> = <expression>), not changing
            ### anything, add the variable to variables_to_solve_for_list and set
            ### instant_update to True
            else:
                variables_to_solve_for_list.append(line.split("=")[0])
                instant_update_list.append(True)
            ### create the sympy equation, move everything on one side (other side = 0)
            eq_sympy_list.append(Eq(0, sympify(right_side) - sympify(left_side)))

        ### 1st find solution of variables for holding v_0
        eq_sympy_list_hold_v_0 = deepcopy(eq_sympy_list)
        for line_idx, line in enumerate(eq_sympy_list_hold_v_0):
            eq_sympy_list_hold_v_0[line_idx] = line.subs(
                {variables_sympy_dict["v"]: variables_sympy_dict["v_0"]}
            )
        ### solve
        solution_hold_v_0 = self._solve_v_clamp_equations(
            eq_sympy_list_hold_v_0, variables_to_solve_for_list, "holding v_0"
        )

        ### 2nd for v_clamp_inst set v to v_step only in equaitons which are
        ### updated instantaneously  (v_clamp and all non-derivatives), for all
        ### derivatives use the solution for holding v_0
        eq_sympy_list_inst = deepcopy(eq_sympy_list)
        for line_idx, line in enumerate(eq_sympy_list_inst):
            if instant_update_list[line_idx]:
                ### variable is updated instantaneously -> set v to v_step
                eq_sympy_list_inst[line_idx] = line.subs(
                    {
                        variables_sympy_dict["v"]: variables_sympy_dict["v_step"],
                    }
                )
            else:
                ### variable is not updated instantaneously -> use solution for hold v_0
                current_variable_name = variables_to_solve_for_list[line_idx]
                current_variable = variables_sympy_dict[current_variable_name]
                eq_sympy_list_inst[line_idx] = Eq(
                    0, solution_hold_v_0[current_variable] - current_variable
                )
        ### solve
        solution_inst = self._solve_v_clamp_equations(
            eq_sympy_list_inst, variables_to_solve_for_list, "step from v_0 to v_step"
        )

        ### 3rd for v_clamp_hold (i.e. holding v_step) set v to v_step in all
        ### equations
        eq_sympy_list_hold = deepcopy(eq_sympy_list)
        for line_idx, line in enumerate(eq_sympy_list_hold):
            eq_sympy_list_hold[line_idx] = line.subs(
                {variables_sympy_dict["v"]: variables_sympy_dict["v_step"]}
            )
        ### solve
        solution_hold = self._solve_v_clamp_equations(
            eq_sympy_list_hold, variables_to_solve_for_list, "holding v_step"
        )

        ### get the equations for v_clamp_inst and v_clamp_hold
        eq_v_clamp_inst = solution_inst[variables_sympy_dict["v_clamp"]]
        eq_v_clamp_hold = solution_hold[variables_sympy_dict["v_clamp"]]
        if self.verbose:
            print(f"Equation for v_clamp_inst: {factor(eq_v_clamp_inst)}")
            print(f"Equation for v_clamp_hold: {factor(eq_v_clamp_hold)}")

        ### create functions for v_clamp_inst and v_clamp_hold
        ### 1st obtain all variables from the equations for v_clamp_inst and v_clamp_hold
        f_variables = list(
            set(list(eq_v_clamp_inst.free_symbols) + list(eq_v_clamp_hold.free_symbols))
        )
        ### 2nd create a function for each equation
        f_inst = lambdify(f_variables, eq_v_clamp_inst)
        f_hold = lambdify(f_variables, eq_v_clamp_hold)

        return f_inst, f_hold, f_variables

    def _solve_v_clamp_equations(
        self, eq_sympy_list, variables_to_solve_for_list, name
    ):
        solution = solve(
            eq_sympy_list,
            variables_to_solve_for_list,
            dict=True,
        )
        if len(solution) == 1:
            solution = solution[0]
        elif len(solution) > 1:
            print(f"Warning: Multiple solutions for {name}!")
        else:
            raise ValueError(f"Could not solve equations for {name}!")

        return solution

    def _get_variables_from_eq(self, eq: str):
        """
        Get a list of all variable names from the given equation string.

        Args:
            eq (str):
                the equation string
        """
        ### split equations into lines
        eq_line_list = eq.splitlines()

        ### loop over lines
        variables_name_list = []
        for line in eq_line_list:
            if "=" not in line:
                continue
            ### split line at = and only take right side (e.g. not use dv/dt)
            line = line.split("=")[1]
            ### remove whitespaces
            line = line.replace(" ", "")
            ### replace all kind of special characters with a space
            special_characters = ["+", "-", "*", "/", "(", ")", "[", "]", "="]
            for special_character in special_characters:
                line = line.replace(special_character, " ")
            ### split line at spaces
            line_split = line.split()
            ### append to list
            variables_name_list += line_split

        ### remove duplicates
        variables_name_list = list(set(variables_name_list))

        return variables_name_list

    def _simulations(self):
        """
        Perform the resting state and voltage step simulations to obtain v_clamp_inst,
        v_clamp_hold and v_rest.

        Returns:
            v_rest (float):
                resting state voltage
            v_clamp_inst (np.array):
                array of the voltage clamp values directly after the voltage step
            v_clamp_hold (np.array):
                array of the voltage clamp values after the holding period

        """
        duration = 200
        ### simulate both models at the same time
        ### for pop_normal nothing happens (resting state)
        ### for pop_clamp the voltage is set to v_0 and then to v_step for each neuron
        get_population("pop_clamp").v = self._v_0_arr
        simulate(duration)
        get_population("pop_clamp").v = self._v_step_arr
        simulate(self._timestep)
        v_clamp_inst_arr = get_population("pop_clamp").v_clamp
        simulate(duration - self._timestep)
        v_clamp_hold_arr = get_population("pop_clamp").v_clamp
        v_rest = get_population("pop_normal").v[0]

        ### get unique values of v_step and their indices
        v_step_unique, v_step_unique_idx = np.unique(
            self._v_step_arr, return_index=True
        )
        ### get the corresponding values of v_clamp_hold (because it does only depend om
        ### v_step)
        v_clamp_hold_unique = v_clamp_hold_arr[v_step_unique_idx]

        if self.do_plot and not isinstance(self._v_clamp_inst_arr, type(None)):
            plt.figure(figsize=(6.4 * 3, 4.8 * 2))
            ### create a 2D color-coded plot of the data for v_clamp_inst and v_clamp_hold
            x = self._v_0_arr
            y = self._v_step_arr

            ### create 2 subplots for original v_clamp_inst and v_clamp_hold
            plt.subplot(231)
            self._plot_v_clamp_subplot(
                x,
                y,
                self._v_clamp_inst_arr,
                "v_clamp_inst original",
            )
            plt.subplot(234)
            self._plot_v_clamp_subplot(
                x,
                y,
                self._v_clamp_hold_arr,
                "v_clamp_hold original",
            )

            ### create 2 subplots for tuned v_clamp_inst and v_clamp_hold
            plt.subplot(232)
            self._plot_v_clamp_subplot(
                x,
                y,
                v_clamp_inst_arr,
                "v_clamp_inst tuned",
            )
            plt.subplot(235)
            self._plot_v_clamp_subplot(
                x,
                y,
                v_clamp_hold_arr,
                "v_clamp_hold tuned",
            )

            ### create 2 subplots for differences
            plt.subplot(233)
            self._plot_v_clamp_subplot(
                x,
                y,
                self._v_clamp_inst_arr - v_clamp_inst_arr,
                "v_clamp_inst diff",
            )
            plt.subplot(236)
            self._plot_v_clamp_subplot(
                x,
                y,
                self._v_clamp_hold_arr - v_clamp_hold_arr,
                "v_clamp_hold diff",
            )

            plt.tight_layout()

            plt.savefig(
                self.plot_file.split(".")[0] + "_data." + self.plot_file.split(".")[1],
                dpi=300,
            )
            plt.close()

        return (
            v_rest,
            v_clamp_inst_arr,
            v_clamp_hold_arr,
            v_step_unique,
            v_clamp_hold_unique,
        )

    def _plot_v_clamp_subplot(self, x, y, c, label):
        plt.title(label)

        ci = c
        if len(c) >= 4:
            # Define the grid for interpolation
            xi, yi = np.meshgrid(
                np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100)
            )

            # Perform the interpolation
            ci = griddata((x, y), c, (xi, yi), method="linear")

            # Plot the interpolated surface
            plt.contourf(
                xi,
                yi,
                ci,
                levels=100,
                cmap="bwr",
                vmin=-af.get_maximum(np.absolute(ci)),
                vmax=af.get_maximum(np.absolute(ci)),
            )

        # Plot also the original data points
        plt.scatter(
            x,
            y,
            c=c,
            cmap="bwr",
            vmin=-af.get_maximum(np.absolute(ci)),
            vmax=af.get_maximum(np.absolute(ci)),
            s=5,
        )

        plt.colorbar(label=label)
        plt.xlabel("v_0")
        plt.ylabel("v_step")

    def _create_voltage_step_arrays(self):
        """
        Create the arrays for the initial voltages and the voltage steps.

        Returns:
            v_0_arr (np.array):
                array of the initial voltages
            v_step_arr (np.array):
                array of the voltage steps

        """
        ### create the unique values of v_step and v_0
        v_0_arr_unique = np.linspace(-90, -40, self.m)
        v_step_arr_unique = np.linspace(-90, -40, self.n)

        ### create a 2D array of all combinations of v_0 and v_step
        v_0_arr = np.repeat(v_0_arr_unique, self.n)
        v_step_arr = np.tile(v_step_arr_unique, self.m)

        return v_0_arr, v_step_arr

    def _create_model(self):
        """
        Create a population (single neuron) for each neuron model.

        Returns:
            model_normal (CompNeuroModel):
                model containing the population with the normal neuron model
            model_clamp (CompNeuroModel):
                model containing the population with the voltage clamped neuron model
        """
        ### setup ANNarchy
        setup(dt=self._timestep, seed=1234)
        ### create a population with the normal neuron model
        model_normal = CompNeuroModel(
            model_creation_function=lambda: Population(
                1, self._neuron_model, name="pop_normal"
            ),
            name="model_normal",
            do_compile=False,
        )
        ### create a population with the voltage clamped neuron model
        model_clamp = CompNeuroModel(
            model_creation_function=lambda: Population(
                len(self._v_0_arr), self._neuron_model_clamp, name="pop_clamp"
            ),
            name="model_clamp",
            compile_folder_name=self.compile_folder_name,
        )

        return model_normal, model_clamp

    def _get_neuron_model_attributes(self, neuron_model: Neuron):
        """
        Get a list of the attributes (parameters and variables) of the given neuron
        model.

        Returns:
            attributes (list):
                list of the attributes of the given neuron model
        """
        neuron_model._analyse()
        attributes = []
        for param in neuron_model.description["parameters"]:
            attributes.append(param["name"])
        for var in neuron_model.description["variables"]:
            attributes.append(var["name"])
        return attributes

    def _get_neuron_model_arguments(self, neuron_model: Neuron):
        """
        Get a dictionary of the initial arguments of the given neuron model.

        Args:
            neuron_model (Neuron):
                the neuron model which should be analyzed

        Returns:
            init_arguments_dict (dict):
                dictionary of the initial arguments of the given neuron model
        """
        ### get the names of the arguments of a Neuron class
        init_arguments_name_list = list(Neuron.__init__.__code__.co_varnames)
        init_arguments_name_list.remove("self")
        init_arguments_name_list.remove("name")
        init_arguments_name_list.remove("description")
        ### get these attributes from the given neuron model
        init_arguments_dict = {
            init_arguments_name: getattr(neuron_model, init_arguments_name)
            for init_arguments_name in init_arguments_name_list
        }

        return init_arguments_dict

    def _get_neuron_model_clamp(self):
        """
        Create a neuron model with voltage clamp equations.

        Returns:
            neuron_model_clamp (Neuron):
                the neuron model with voltage clamped equation
        """
        ### get these attributes from the given neuron model
        init_arguments_dict = self._get_neuron_model_arguments(self._neuron_model)
        ### split the equations string
        equations_line_split_list = str(init_arguments_dict["equations"]).splitlines()
        ### adjust the equations for voltage clamp
        equations_line_split_list = self._adjust_equations_for_voltage_clamp(
            equations_line_split_list
        )

        ### combine string lines to multiline strings again
        init_arguments_dict["equations"] = "\n".join(equations_line_split_list)

        ### create neuron model with new equations
        neuron_model_clamp = Neuron(**init_arguments_dict)

        if self.verbose:
            print(f"Neuron model with voltage clamp equations:\n{neuron_model_clamp}")

        return neuron_model_clamp

    def _adjust_equations_for_voltage_clamp(self, eq_line_list: list):
        """
        Replaces the 'dv/dt' or 'v+=' equation with a voltage clamp version in which the
        new variable 'v_clamp' is calculated from the right side of the 'dv/dt' or 'v+='
        equation.

        Args:
            eq_line_list (list):
                list of the lines of the equations of the neuron model

        Returns:
            eq_line_list (list):
                list of the lines of the equations of the neuron model with voltage clamp
        """
        ### check in which lines v is updated
        line_is_v_list = [False] * len(eq_line_list)
        for line_idx, line in enumerate(eq_line_list):
            line_is_v_list[line_idx] = self._get_line_is_v(line)
        ### raise error if in no line v is updated or in multiple lines
        if sum(line_is_v_list) == 0 or sum(line_is_v_list) > 1:
            raise ValueError(
                "Could not find one line with dv/dt or v+= in equations of neuronmodel!"
            )

        ### obtain the line containing v update
        eq_v = eq_line_list[line_is_v_list.index(True)]

        ### remove whitespaces
        eq_v = eq_v.replace(" ", "")

        ### split eqatuion at ":" to separate flags
        eq_v_split = eq_v.split(":")
        eq_v = eq_v_split[0]
        ### check if flags are present
        if len(eq_v_split) == 1:
            flags = ""
        else:
            flags = ":" + eq_v_split[1]
        ### adjust the equation for voltage clamp
        if "+=" in eq_v:
            eq_v, eq_v_clamp = self._adjust_equation_for_voltage_clamp_plus(eq_v, flags)
        else:
            eq_v, eq_v_clamp = self._adjust_equation_for_voltage_clamp_dvdt(eq_v, flags)
        ### delete old equation from equation list using the index of the equation
        eq_line_list.pop(line_is_v_list.index(True))
        ### insert new equation at the same position
        eq_line_list.insert(line_is_v_list.index(True), eq_v)
        ### insert new equation for "v_clamp" at the same position
        eq_line_list.insert(line_is_v_list.index(True), eq_v_clamp)

        return eq_line_list

    def _adjust_equation_for_voltage_clamp_plus(self, eq_v: str, flags: str):
        """
        Convert the v-update equation using "v+=" into a voltage clamp version.

        Args:
            eq_v (str):
                the equation string for updating v (without flags)
            flags (str):
                the flags of the equation string

        Returns:
            eq_v (str):
                the adjusted equation string for updating v (without flags)
            eq_v_clamp (str):
                the equation string for "v_clamp" (with flags)
        """
        ### split equations at "=" to separate left and right side
        eq_v_left, eq_v_right = eq_v.split("=")
        ### set right side to zero and combine equation again with "="
        eq_v = eq_v_left + "=" + "0"
        ### create new equation for "v_clamp" with right side of original equation
        eq_v_clamp = "v_clamp=" + eq_v_right + flags

        return eq_v, eq_v_clamp

    def _adjust_equation_for_voltage_clamp_dvdt(self, eq_v: str, flags: str):
        """
        Convert the v-update equation using "dv/dt" into a voltage clamp version.

        Args:
            eq_v (str):
                the equation string for updating v (without flags)
            flags (str):
                the flags of the equation string

        Returns:
            eq_v (str):
                the adjusted equation string for updating v (without flags)
            eq_v_clamp (str):
                the equation string for "v_clamp" (with flags)
        """
        ### if equation starts with "dv/dt=" do the same as for "v+="
        if eq_v.startswith("dv/dt="):
            return self._adjust_equation_for_voltage_clamp_plus(eq_v, flags)

        ### if equation doesn't start with "dv/dt=" --> need to rearrange equation
        ### i.e. solve the equation for dv/dt
        eq_v = eq_v.replace("dv/dt", "delta_v")

        ### split the equation at "=" and move everything on one side (other side = 0)
        ### replace the whole right side with "right_side" making solving easier
        left_side, right_side = eq_v.split("=")
        eq_v_one_side = f"(right_side) - {left_side}"

        ### prepare the sympy equation generation
        attributes_name_list = self._get_neuron_model_attributes(self._neuron_model)
        ### create a sympy symbol for each attribute of the neuron
        attributes_tuple = symbols(",".join(attributes_name_list))
        ### create a dict with the names as keys and the sympy symbols as values
        attributes_sympy_dict = {
            key: attributes_tuple[attributes_name_list.index(key)]
            for key in attributes_name_list
        }
        ### further create symbols for delta_v and right_side
        attributes_sympy_dict["delta_v"] = Symbol("delta_v")
        attributes_sympy_dict["right_side"] = Symbol("right_side")

        ### now creating the sympy equation
        eq_sympy = sympify(eq_v_one_side)

        ### solve the equation for delta_v
        result = solve(eq_sympy, attributes_sympy_dict["delta_v"], dict=True)
        if len(result) != 1:
            raise ValueError("Could not solve equation of neuronmodel for dv/dt!")

        ### convert result to string
        result = str(result[0][attributes_sympy_dict["delta_v"]])

        ### replace "right_side" by the actual right side in brackets
        result = result.replace("right_side", f"({right_side})")

        ### create new equation for dv/dt
        eq_v = "dv/dt = 0"
        ### create new equation for "v_clamp" with the equation solved for dv/dt
        eq_v_clamp = "v_clamp=" + result + flags

        return eq_v, eq_v_clamp

    def _get_line_is_v(self, line: str):
        """
        Check if a equation string contains dv/dt or v+=

        Args:
            line (str):
                the equation string

        Returns:
            line_is_v (bool):
                True if the equation string contains dv/dt or v+=, False otherwise
        """
        if "v" not in line:
            return False

        ### remove whitespaces
        line = line.replace(" ", "")

        ### check for dv/dt
        if "dv/dt" in line:
            return True

        ### check for v update
        if "v+=" in line and line.startswith("v"):
            return True

        return False


def interactive_plot(
    nrows: int,
    ncols: int,
    sliders: list[dict],
    create_plot: Callable,
):
    """
    Create an interactive plot with sliders.

    Args:
        nrows (int):
            number of rows of subplots
        ncols (int):
            number of columns of subplots
        sliders (list):
            list of dictionaries with slider kwargs (see matplotlib.widgets.Slider), at
            least the following keys have to be present:
                - label (str):
                    label of the slider
                - valmin (float):
                    minimum value of the slider
                - valmax (float):
                    maximum value of the slider
        create_plot (Callable):
            function which fills the subplots, has to have the signature
            create_plot(axs, sliders), where axs is a list of axes (for each subplot)
            and sliders is the given sliders list with newly added keys "ax" (axes of
            the slider) and "slider" (the Slider object itself, so that you can access
            the slider values in the create_plot function using the .val attribute)

    Examples:
        ```python
        def create_plot(axs, sliders):
            axs[0].axhline(sliders[0]["slider"].val, color="r")
            axs[1].axvline(sliders[1]["slider"].val, color="r")

        interactive_plot(
            nrows=2,
            ncols=1,
            sliders=[
                {"label": "a", "valmin": 0.0, "valmax": 1.0, "valinit": 0.3},
                {"label": "b", "valmin": 0.0, "valmax": 1.0, "valinit": 0.7},
            ],
            create_plot=create_plot,
        )
        ```
    """

    def update(axs, sliders):
        ### remove everything from all axes except the sliders axes
        for ax in axs:
            if ax not in [slider["ax"] for slider in sliders]:
                ax.cla()
        ### recreate the plot
        create_plot(axs, sliders)
        ### redraw the canvas
        fig.canvas.draw_idle()

    ### create the figure as large as the screen
    screen_width, screen_height = get_monitors()[0].width, get_monitors()[0].height
    figsize = (screen_width / 100, screen_height / 100)
    fig = plt.figure(figsize=figsize)

    ### create the axes filled with the create_plot function
    grid = GridSpec((nrows + 1) * len(sliders), ncols * len(sliders), figure=fig)
    axs = []
    for row_idx in range(nrows):
        for col_idx in range(ncols):
            ax = fig.add_subplot(
                grid[
                    row_idx * len(sliders) : (row_idx + 1) * len(sliders),
                    col_idx * len(sliders) : (col_idx + 1) * len(sliders),
                ]
            )
            axs.append(ax)

    ### create the sliders axes
    for slider_idx, slider_kwargs in enumerate(sliders):
        sliders[slider_idx]["ax"] = fig.add_subplot(
            grid[nrows * len(sliders) + slider_idx, :]
        )

    ### initialize the sliders to their axes
    for slider_idx, slider_kwargs in enumerate(sliders):
        ### if init out of min max, change min max
        if "valinit" in slider_kwargs:
            if slider_kwargs["valinit"] < slider_kwargs["valmin"]:
                slider_kwargs["valmin"] = slider_kwargs["valinit"]
            elif slider_kwargs["valinit"] > slider_kwargs["valmax"]:
                slider_kwargs["valmax"] = slider_kwargs["valinit"]
        slider = Slider(**slider_kwargs)
        slider.on_changed(lambda val: update(axs, sliders))
        sliders[slider_idx]["slider"] = slider

    ### create the plot
    create_plot(axs, sliders)
    ### arange subplots
    plt.tight_layout()
    new_right_border = 0.85
    new_left_border = 0.15
    for slider_idx, slider_kwargs in enumerate(sliders):
        ax = sliders[slider_idx]["ax"]
        ### set new borders
        ax.set_position(
            [
                new_left_border,
                ax.get_position().y0,
                new_right_border - new_left_border,
                ax.get_position().height,
            ]
        )

    ### show the plot
    plt.show()


def efel_loss(trace1, trace2, feature_list):
    """
    Calculate the loss between two traces using the features from the feature_list.

    Args:
        trace1 (dict):
            dictionary with the keys "T" (time), "V" (voltage), "stim_start" (start of
            the stimulus), "stim_end" (end of the stimulus)
        trace2 (dict):
            dictionary with the keys "T" (time), "V" (voltage), "stim_start" (start of
            the stimulus), "stim_end" (end of the stimulus)
        feature_list (list):
            list of feature names which should be used to calculate the loss (see
            https://efel.readthedocs.io/en/latest/eFeatures.html, some of them are
            available)

    Returns:
        loss (np.array):
            array with the loss
    """
    verbose = False
    ### set a plausible "maximum" absolute difference for each feature
    diff_max = {
        "steady_state_voltage_stimend": 200,
        "steady_state_voltage": 200,
        "voltage_base": 200,
        "voltage_after_stim": 200,
        "minimum_voltage": 200,
        "time_to_first_spike": trace1["T"][-1] - trace1["stim_start"][0],
        "time_to_second_spike": trace1["T"][-1] - trace1["stim_start"][0],
        "time_to_last_spike": trace1["T"][-1] - trace1["stim_start"][0],
        "spike_count": len(trace1["T"]),
        "spike_count_stimint": len(
            trace1["T"][
                (
                    (trace1["T"] >= trace1["stim_start"][0]).astype(int)
                    * (trace1["T"] < trace1["stim_end"][0]).astype(int)
                ).astype(bool)
            ]
        ),
        "ISI_CV": 1,
    }
    if verbose:
        print(f"\ndiff_max: {diff_max}")

    ### set a plausible "close" absolute difference for each feature
    diff_close = {
        "steady_state_voltage_stimend": 10,
        "steady_state_voltage": 10,
        "voltage_base": 10,
        "voltage_after_stim": 10,
        "minimum_voltage": 10,
        "time_to_first_spike": np.clip(
            (trace1["T"][-1] - trace1["stim_start"][0]) * 0.1, 5, 50
        ),
        "time_to_second_spike": np.clip(
            (trace1["T"][-1] - trace1["stim_start"][0]) * 0.1, 5, 50
        ),
        "time_to_last_spike": np.clip(
            (trace1["T"][-1] - trace1["stim_start"][0]) * 0.1, 5, 50
        ),
        "spike_count": np.ceil((trace1["T"][-1] - trace1["T"][0]) / 200),
        "spike_count_stimint": np.ceil((trace1["T"][-1] - trace1["T"][0]) / 200),
        "ISI_CV": 0.1,
    }
    if verbose:
        print(f"\ndiff_close: {diff_close}\n")

    ### catch if features from feature_list are not supported
    features_not_supported = [
        feature for feature in feature_list if feature not in diff_max
    ]
    if features_not_supported:
        raise ValueError(f"Features not supported: {features_not_supported}")

    ### catch "exploding" neurons by returning max loss of features
    if (
        np.any(trace1["V"] < -200)
        or np.any(trace1["V"] > 100)
        or np.any(trace2["V"] < -200)
        or np.any(trace2["V"] > 100)
    ):
        loss = 0
        for feature in feature_list:
            loss += diff_max[feature] / diff_close[feature]
        loss /= len(feature_list)
        loss = np.array([loss])
        if verbose:
            print(f"loss: {loss}")
        return loss

    ### calculate and return the mean of the differences of the features
    features_1, features_2 = efel.getFeatureValues(
        [trace1, trace2],
        feature_list,
        raise_warnings=False,
    )
    if verbose:
        print(f"\nfeatures_1: {features_1}\n")
        print(f"features_2: {features_2}\n")
    loss = 0
    for feature in feature_list:
        ### if both features are None use 0
        if features_1[feature] is None and features_2[feature] is None:
            diff = 0
        ### if single feature is None use diff_max
        elif features_1[feature] is None or features_2[feature] is None:
            diff = diff_max[feature]
        ### if features contain multiple values use the mean TODO not tested yet
        elif len(features_1[feature]) > 1 or len(features_2[feature]) > 1:
            if verbose:
                print("features with multiple values not tested yet!")
            diff = np.mean(
                np.absolute(features_1[feature] - features_2[feature]), keepdims=True
            )
        else:
            diff = np.absolute(features_1[feature] - features_2[feature])
        ### scale the difference by diff_close and add to loss
        loss += diff / diff_close[feature]
    loss /= len(feature_list)

    if verbose:
        print(f"loss: {loss}")
    return loss
