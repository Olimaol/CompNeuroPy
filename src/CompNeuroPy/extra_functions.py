import pandas as pd
from contextlib import contextmanager
import sys
import os
from CompNeuroPy import analysis_functions as af
from CompNeuroPy import system_functions as sf
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


def print_df(df):
    """
    Prints the complete dataframe df

    Args:
        df (pandas dataframe):
            Dataframe to be printed
    """
    with pd.option_context(
        "display.max_rows", None
    ):  # more options can be specified also
        print(df)


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


def _prepare_cma_deap(
    lower: np.ndarray, upper: np.ndarray, evaluate_function: Callable, param_names=None
):
    """
    Prepares the deap Covariance Matrix Adaptation Evolution Strategy optimization.

    Args:
        lower (np.ndarray):
            Lower bounds of the parameters
        upper (np.ndarray):
            Upper bounds of the parameters
        evaluate_function (Callable):
            Function evaluating the losses of a population of individuals. Should be
            a list of tuples with the losses of the individuals.

    Returns:
        deap_dict (dict):
            Dictionary containing the deap toolbox, hall of fame, statistics, lower
            and upper bounds
        lambda_ (int):
            Number of individuals in a population
    """
    ### create scaler to scale parameters into range [0,1] based on lower and upper bounds
    upper_orig = deepcopy(upper)
    lower_orig = deepcopy(lower)
    scaler = lambda x: (x - lower_orig) / (upper_orig - lower_orig)

    ### create inverse scaler to scale parameters back into original range [lower,upper]
    inv_scaler = lambda x: x * (upper_orig - lower_orig) + lower_orig

    ### scale upper and lower bounds
    lower = scaler(lower)
    upper = scaler(upper)

    ### create the individual class
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    ### create the toolbox
    toolbox = base.Toolbox()
    ### function calculating losses from individuals (from whole population)
    toolbox.register("evaluate", evaluate_function)
    ### search strategy
    strategy = cma.Strategy(
        centroid=(lower + upper) / 2,
        sigma=upper - lower,
    )
    strategy.ccov1 *= 0.5
    strategy.ccovmu *= 0.5
    # strategy.damps /= 0.5
    print(
        f"lambda (The number of children to produce at each generation): {strategy.lambda_}"
    )
    print(f"mu (The number of parents to keep from the lambda children): {strategy.mu}")
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
    }, strategy.lambda_


def _cma_deap(max_evals, deap_plot_file, deap_dict):
    """
    Runs the optimization with deap.

    Args:
        max_evals (int):
            number of runs (here generations) the optimization method performs

        deap_plot_file (str):
            the name of the figure which will be saved and shows the logbook
    """
    ### run the search algorithm with the prepared deap_dict
    pop, logbook = _deap_ea_generate_update(
        toolbox=deap_dict["toolbox"],
        ngen=max_evals,
        lower=deap_dict["lower"],
        upper=deap_dict["upper"],
        inv_scaler=deap_dict["inv_scaler"],
        stats=deap_dict["stats"],
        halloffame=deap_dict["hof"],
        verbose=False,
    )

    ### scale parameters of hall of fame back into original range [lower,upper]
    hof_final = deap_dict["inv_scaler"](deap_dict["hof"][0])

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
    sf.create_dir("/".join(deap_plot_file.split("/")[:-1]))
    fig.savefig(deap_plot_file, dpi=300)

    return best


def _deap_ea_generate_update(
    toolbox,
    ngen,
    lower,
    upper,
    inv_scaler,
    halloffame=None,
    stats=None,
    verbose=__debug__,
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
        toolbox:
            A deap Toolbox object that contains the evolution operators.
        ngen:
            The number of generations to run.
        lower:
            A list of lower bounds for the individuals.
        upper:
            A list of upper bounds for the individuals.
        inv_scaler:
            A function to scale parameters back into original range [lower,upper]
        halloffame:
            A deap HallOfFame object that will to track the best individuals
        stats:
            A deap Statistics object to track the statistics of the evolution.
        verbose:
            Whether or not to print the statistics for each gen.

    Returns:
        population:
            A list of individuals.
        logbook:
            A Logbook() object that contains the evolution statistics.
    """
    ### init logbook
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    ### define progress bar
    progress_bar = tqdm(range(ngen), total=ngen, unit="gen")

    ### loop over generations
    for gen in progress_bar:
        ### Generate a new population
        population = toolbox.generate()
        ### clip individuals of population to variable bounds
        for ind in population:
            for idx in range(len(ind)):
                if ind[idx] < lower[idx]:
                    ind[idx] = lower[idx]
                elif ind[idx] > upper[idx]:
                    ind[idx] = upper[idx]
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

        ### Append the current generation statistics to the logbook
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        if verbose:
            print(logbook.stream)

        ### update progress bar with current best loss
        progress_bar.set_postfix_str(
            f"best loss: {halloffame[0].fitness.values[0]:.5f}"
        )

    return population, logbook
