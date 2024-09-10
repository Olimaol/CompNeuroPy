import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from CompNeuroPy import (
    DeapCma,
    save_variables,
    load_variables,
    run_script_parallel,
    create_data_raw_folder,
    create_dir,
    RNG,
)
from scipy.optimize import minimize, Bounds
from scipy.interpolate import griddata
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from scipy.optimize import curve_fit

from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import itertools


def gauss_1d(x, amp, mean, sig):
    return amp * np.exp(-((x - mean) ** 2) / (2 * sig**2))


def log_normal_1d(x, amp, mean, sig):
    return (amp / x) * np.exp(-((np.log(x) - mean) ** 2) / (2 * sig**2))


def regression_func(
    X,
    denormalize: None | str,
    args: list,
):
    """
    A 2D regression function.

    Args:
        X (array):
            The x (X[0]) and y (X[1]) coordinates. Needs to be normalized.
        denormalize (None | str):
            The variable name to denormalize the calculated values.
        args (list):
            The parameters of the regression function.

    Returns:
        float:
            The z(x,y) value(s) of the regression function.
    """
    x, y = X

    ### 2D polynomial with certain degree
    ret = np.clip(
        args[0]
        + gauss_1d(x, amp=args[1], mean=args[2], sig=args[3])
        + gauss_1d(y, amp=args[4], mean=args[5], sig=args[6])
        + gauss_1d(x * y, amp=args[7], mean=args[8], sig=args[9])
        + gauss_1d(x**2 * y, amp=args[10], mean=args[11], sig=args[12])
        + gauss_1d(x * y**2, amp=args[13], mean=args[14], sig=args[15]),
        -1e20,
        1e20,
    )

    ### denormalize the calculated values, during regression optimization the target
    ### values are normalized
    if denormalize is not None:
        ret = post_process_for_regression(var_value=ret, var_name=denormalize)

    return ret


def plot_2d_interpolated_image(
    x, y, z, vmin=None, vmax=None, grid_size=100, method="linear", cmap="viridis"
):
    """
    Plots a 2D color-coded image of "D grid data.

    Args:
        x (array):
            The x-coordinates of the data points.
        y (array):
            The y-coordinates of the data points.
        z (array):
            The z-values corresponding to the (x, y) coordinates.
        vmin (float, optional):
            The minimum value for the color scale. If not provided, the minimum value of
            z is used.
        vmax (float):
            The maximum value for the color scale. If not provided, the maximum value of
            z is used.
        grid_size (int):
            The size of the grid for plotting (default: 100).
        method (str):
            The interpolation method to use (default: "linear").
    """
    # Define the grid for interpolation
    xi = np.linspace(min(x), max(x), grid_size)
    yi = np.linspace(min(y), max(y), grid_size)
    xi, yi = np.meshgrid(xi, yi)

    # Perform the interpolation
    zi = griddata((x, y), z, (xi, yi), method=method)

    # print(f"max interpolated: {np.max(zi)}")

    # Plot the interpolated data
    if vmin is None:
        vmin = np.min(z)
    if vmax is None:
        vmax = np.max(z)
    plt.contourf(xi, yi, zi, levels=100, cmap=cmap, vmin=vmin, vmax=vmax)

    # plot scatter plot of original data
    plt.scatter(x, y, c=z, cmap=cmap, vmin=vmin, vmax=vmax, edgecolor="k", marker="o")


def generate_samples(n, p, mean_shift=0, std_scale=1):
    """
    Generate samples of a binomial and a normal distribution. The normal distribution is
    generated to best approximate the binomial distribution. Further the normal
    distribution is shifted by mean_shift and scaled by std_scale.

    Args:
        n (int):
            The number of trials of the binomial distribution.
        p (float):
            The probability of success of the binomial distribution.
        mean_shift (float):
            The shift of the mean of the normal distribution.
        std_scale (float):
            The scaling of the standard deviation of the normal distribution.
    """
    # Generate data samples
    binomial_sample = RNG(seed=SEED).rng.binomial(n, p, S)
    mean = n * p
    std_dev = np.sqrt(n * p * (1 - p))
    normal_sample = RNG(seed=SEED).rng.normal(mean + mean_shift, std_dev * std_scale, S)

    ### round and clip the normal sample
    normal_sample = np.round(normal_sample)
    normal_sample[normal_sample < 0] = 0
    normal_sample[normal_sample > n] = n

    return binomial_sample, normal_sample


def get_difference_of_samples(binomial_sample, normal_sample, n):
    """
    Calculate the difference between samples of a binomial and a normal distribution.

    Args:
        binomial_sample (array):
            The sample of the binomial distribution.
        normal_sample (array):
            The sample of the normal distribution.
        n (int):
            The number of trials of the binomial distribution.
    """
    diff = (
        np.histogram(binomial_sample, bins=n + 1, range=(-0.5, n + 0.5))[0]
        - np.histogram(normal_sample, bins=n + 1, range=(-0.5, n + 0.5))[0]
    )
    return np.sum(np.abs(diff)) / (2 * len(binomial_sample))


def difference_binomial_normal_optimize(n, p):
    """
    Calculate the difference between samples of a binomial and a normal distribution.
    The binomial distribution is generated with parameters n and p.
    The normal distribution is generated to best approximate the binomial distribution.
    Further the normal distribution is shifted by mean_shift and scaled by std_scale.
    Both are optimized to minimize the difference between the binomial and normal
    distribution.

    Args:
        n (int):
            The number of trials of the binomial distribution.
        p (float):
            The probability of success of the binomial distribution.

    Returns:
        mean_shift_opt (float):
            The shift of the mean of the normal distribution.
        std_scale_opt (float):
            The scaling of the standard deviation of the normal distribution.
        error_opt (float):
            The difference between the binomial and normal distribution.
    """
    ### save p and n to be availyble in optimization script
    save_variables(
        variable_list=[p, n],
        name_list=["p", "n"],
        path=OPTIMIZE_FOLDER,
    )
    ### run optimization
    args_list = [[f"{parallel_id}"] for parallel_id in range(N_RUNS_OPT_PER_PAIR)]
    run_script_parallel(
        script_path="test2_deap_opt_transform.py",
        n_jobs=N_JOBS,
        args_list=args_list,
    )
    ### get best mean_shift and std_scale by loading all optimizations and check best
    ### fitness
    best_fitness = 1e6
    best_parallel_id = 0
    for parallel_id in range(N_RUNS_OPT_PER_PAIR):
        loaded_variables = load_variables(
            name_list=[
                f"error_opt_{parallel_id}",
            ],
            path=OPTIMIZE_FOLDER,
        )
        error_opt = loaded_variables[f"error_opt_{parallel_id}"]
        if error_opt < best_fitness:
            best_fitness = error_opt
            best_parallel_id = parallel_id

    loaded_variables = load_variables(
        name_list=[
            f"mean_shift_opt_{best_parallel_id}",
            f"std_scale_opt_{best_parallel_id}",
            f"error_opt_{best_parallel_id}",
        ],
        path=OPTIMIZE_FOLDER,
    )
    mean_shift_opt = loaded_variables[f"mean_shift_opt_{best_parallel_id}"]
    std_scale_opt = loaded_variables[f"std_scale_opt_{best_parallel_id}"]
    error_opt = loaded_variables[f"error_opt_{best_parallel_id}"]

    return mean_shift_opt, std_scale_opt, error_opt


def difference_binomial_normal_mixed(n, p):
    """
    Calculate the difference between samples of a binomial and a normal distribution.
    The binomial distribution is generated with parameters n and p.
    The normal distribution is generated to best approximate the binomial distribution.
    Further the normal distribution is shifted by mean_shift and scaled by std_scale.
    Both are obtained either from optimization, regression or self-defined values.

    Args:
        n (int):
            The number of trials of the binomial distribution.
        p (float):
            The probability of success of the binomial distribution.

    Returns:
        mean_shift_mixed (float):
            The shift of the mean of the normal distribution.
        std_scale_mixed (float):
            The scaling of the standard deviation of the normal distribution.
        error_mixed (float):
            The difference between the binomial and normal distribution.
    """
    ### load parameters for regression
    loaded_variables = load_variables(
        name_list=[
            "popt_mean_shift",
            "popt_std_scale",
        ],
        path=REGRESS_FOLDER,
    )
    ### regression was optimized with normalized data thus need to normalize the data
    ### here too and after regression denormalize the results
    n = preprocess_for_regress(var_value=n, var_name="n")
    p = preprocess_for_regress(var_value=p, var_name="p")
    std_scale_regress = regression_func(
        X=(n, p), denormalize="std_scale", args=loaded_variables["popt_std_scale"]
    )
    n = post_process_for_regression(var_value=n, var_name="n")
    p = post_process_for_regression(var_value=p, var_name="p")

    ### for mixed version only use regressed std_scale
    mean_shift_mixed = 0
    std_scale_mixed = std_scale_regress
    error_mixed = difference_binomial_normal(
        mean_shift=mean_shift_mixed, std_scale=std_scale_mixed, n=n, p=p
    )

    return mean_shift_mixed, std_scale_mixed, error_mixed


def difference_binomial_normal_regress(n, p):
    """
    Calculate the difference between samples of a binomial and a normal distribution.
    The binomial distribution is generated with parameters n and p.
    The normal distribution is generated to best approximate the binomial distribution.
    Further the normal distribution is shifted by mean_shift and scaled by std_scale.
    Both are obtained from the regression of the optimized mean_shift and std_scale.

    Args:
        n (int):
            The number of trials of the binomial distribution.
        p (float):
            The probability of success of the binomial distribution.

    Returns:
        mean_shift_regress (float):
            The shift of the mean of the normal distribution.
        std_scale_regress (float):
            The scaling of the standard deviation of the normal distribution.
        error_regress (float):
            The difference between the binomial and normal distribution.
    """
    ### load parameters for regression
    loaded_variables = load_variables(
        name_list=[
            "popt_mean_shift",
            "popt_std_scale",
        ],
        path=REGRESS_FOLDER,
    )
    ### regression was optimized with normalized data thus need to normalize the data
    ### here too and after regression denormalize the results
    n = preprocess_for_regress(var_value=n, var_name="n")
    p = preprocess_for_regress(var_value=p, var_name="p")
    mean_shift_regress = regression_func(
        X=(n, p), denormalize="mean_shift", args=loaded_variables["popt_mean_shift"]
    )
    std_scale_regress = regression_func(
        X=(n, p), denormalize="std_scale", args=loaded_variables["popt_std_scale"]
    )
    n = post_process_for_regression(var_value=n, var_name="n")
    p = post_process_for_regression(var_value=p, var_name="p")
    error_regress = difference_binomial_normal(
        mean_shift=mean_shift_regress, std_scale=std_scale_regress, n=n, p=p
    )

    return mean_shift_regress, std_scale_regress, error_regress


def difference_binomial_normal(mean_shift, std_scale, n, p):
    """
    Calculate the difference between samples of a binomial and a normal distribution.
    The binomial distribution is generated with parameters n and p.
    The normal distribution is generated to best approximate the binomial distribution.
    Further the normal distribution is shifted by mean_shift and scaled by std_scale.

    Args:
        mean_shift (float):
            The shift of the mean of the normal distribution.
        std_scale (float):
            The scaling of the standard deviation of the normal distribution.
        n (int):
            The number of trials of the binomial distribution.
        p (float):
            The probability of success of the binomial distribution.

    Returns:
        diff (float):
            The difference between the binomial and normal distribution.
    """
    # Generate data samples
    binomial_sample, normal_sample = generate_samples(
        n=n, p=p, mean_shift=mean_shift, std_scale=std_scale
    )

    # Calculate difference
    diff = get_difference_of_samples(binomial_sample, normal_sample, n=n)
    return diff


def logarithmic_arange(start, end, num_points):
    """
    Generate a list of logarithmically spaced points between a start and end point. The
    smaller side of the range is denser with points.

    Args:
        start (float):
            The starting point of the distribution.
        end (float):
            The ending point of the distribution.
        num_points (int):
            The number of points to generate.

    Returns:
        points (list):
            A list of logarithmically spaced points.
    """
    if start <= 0 or end <= 0:
        raise ValueError("Start and end points must be positive numbers.")
    if num_points < 2:
        raise ValueError("Number of points must be at least 2.")

    # Create an array of logarithmically spaced points
    log_start = np.log10(start)
    log_end = np.log10(end)
    log_points = np.linspace(log_start, log_end, num_points)
    points = np.power(10, log_points)

    return points


def plot_with_transformation(mode: str):
    """
    Plot the difference between the binomial and normal distribution for various n and
    p values. Further plots the optimized mean_shift and std_scale values. Load the data
    from the OPTIMIZE_FOLDER and save the plot to the PLOTS_FOLDER.

    Args:
        mode (str):
            Either 'opt', 'regress', or 'mixed'.
    """
    possible_modes = ["opt", "regress", "mixed"]
    if mode not in possible_modes:
        raise ValueError("Mode must be either 'opt', 'regress', or 'mixed'.")

    ### load the data
    loaded_variables = load_variables(
        name_list=[
            "p_list",
            "n_list",
            f"mean_shift_{mode}_list",
            f"std_scale_{mode}_list",
            f"diff_{mode}_list",
        ],
        path=[OPTIMIZE_FOLDER, REGRESS_FOLDER, MIXED_FOLDER][
            possible_modes.index(mode)
        ],
    )
    ### plot the data
    plt.figure(figsize=(6.4 * 2, 4.8 * 2 * 3))
    for idx, title, key in [
        (
            1,
            ["Mean Shift optimized", "Mean Shift regressed", "Mean Shift mixed"][
                possible_modes.index(mode)
            ],
            f"mean_shift_{mode}_list",
        ),
        (
            2,
            ["Std Scale optimized", "Std Scale regressed", "Std Scale mixed"][
                possible_modes.index(mode)
            ],
            f"std_scale_{mode}_list",
        ),
        (
            3,
            ["Difference optimized", "Difference regressed", "Difference mixed"][
                possible_modes.index(mode)
            ],
            f"diff_{mode}_list",
        ),
    ]:
        plt.subplot(3, 1, idx)
        vmin = {
            f"mean_shift_{mode}_list": -np.max(
                np.absolute(np.array(loaded_variables[key]))
            ),
            f"std_scale_{mode}_list": 1
            - np.max(np.absolute(np.array(loaded_variables[key]) - 1)),
            f"diff_{mode}_list": 0,
        }[key]
        vmax = {
            f"mean_shift_{mode}_list": np.max(
                np.absolute(np.array(loaded_variables[key]))
            ),
            f"std_scale_{mode}_list": 1
            + np.max(np.absolute(np.array(loaded_variables[key]) - 1)),
            f"diff_{mode}_list": np.max(loaded_variables[key]),
        }[key]
        cmap = {
            f"mean_shift_{mode}_list": "coolwarm",
            f"std_scale_{mode}_list": "coolwarm",
            f"diff_{mode}_list": "viridis",
        }[key]
        plot_2d_interpolated_image(
            x=loaded_variables["n_list"],
            y=loaded_variables["p_list"],
            z=loaded_variables[key],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        plt.colorbar()
        plt.xlabel("n")
        plt.ylabel("p")
        plt.title(
            f"{title}\n(min: {np.min(loaded_variables[key])}, max: {np.max(loaded_variables[key])})"
        )
    plt.tight_layout()
    create_dir(PLOTS_FOLDER)
    plt.savefig(
        (
            f"{PLOTS_FOLDER}/difference_{['optimized', 'regressed', 'mixed'][possible_modes.index(mode)]}.png"
        ),
        dpi=300,
    )


def plot_compare_original():
    """
    Plot the difference between the binomial and normal distribution for various n and
    p values. Load the data from the COMPARE_ORIGINAL_FOLDER and save the plot to the
    PLOTS_FOLDER.
    """
    ### load the data
    loaded_variables = load_variables(
        name_list=[
            "p_list",
            "n_list",
            "diff_list",
        ],
        path=COMPARE_ORIGINAL_FOLDER,
    )
    ### plot the data
    plt.figure(figsize=(6.4 * 2, 4.8 * 2))
    plt.subplot(1, 1, 1)
    plot_2d_interpolated_image(
        x=loaded_variables["n_list"],
        y=loaded_variables["p_list"],
        z=loaded_variables["diff_list"],
        vmin=0,
        vmax=np.max(loaded_variables["diff_list"]),
        cmap="viridis",
    )
    plt.colorbar()
    plt.xlabel("n")
    plt.ylabel("p")
    plt.title(
        f"Difference original\n(min: {np.min(loaded_variables['diff_list'])}, max: {np.max(loaded_variables['diff_list'])})"
    )
    plt.tight_layout()
    create_dir(PLOTS_FOLDER)
    plt.savefig(f"{PLOTS_FOLDER}/difference_original.png", dpi=300)


def compare_normal_binomial(compare_original, optimize, regress, mixed):
    """
    Compare the difference between the binomial and normal distribution for various n and
    p values with and without optimization or regression.
    """
    if not compare_original and not optimize and not regress and not mixed:
        return

    ### create the n/p pairs
    n_arr = logarithmic_arange(*N_VALUES).astype(int)
    p_arr = logarithmic_arange(*P_VALUES)
    ### get all possible combinations of n and p
    np_pair_arr = list(itertools.product(n_arr, p_arr))

    ### get difference between binomial and normal distribution for each n/p pair
    p_list = []
    n_list = []
    diff_original_list = []
    mean_shift_opt_list = []
    std_scale_opt_list = []
    diff_opt_list = []
    mean_shift_regress_list = []
    std_scale_regress_list = []
    diff_regress_list = []
    mean_shift_mixed_list = []
    std_scale_mixed_list = []
    diff_mixed_list = []
    progress_bar = tqdm(
        np_pair_arr,
        desc=f"Compare {['','original'][int(compare_original)]} {['','optimized'][int(optimize)]} {['','regression'][int(regress)]} {['','mixed'][int(mixed)]}",
    )
    for n, p in progress_bar:
        p_list.append(p)
        n_list.append(n)
        if compare_original:
            ### get the error without optimization
            error = difference_binomial_normal(mean_shift=0, std_scale=1, n=n, p=p)
            diff_original_list.append(error)
        if optimize:
            ### get the error with optimization
            mean_shift_opt, std_scale_opt, error_opt = (
                difference_binomial_normal_optimize(n=n, p=p)
            )
            mean_shift_opt_list.append(mean_shift_opt)
            std_scale_opt_list.append(std_scale_opt)
            diff_opt_list.append(error_opt)
        if regress:
            ### get the error with the regression
            mean_shift_regress, std_scale_regress, error_regress = (
                difference_binomial_normal_regress(n=n, p=p)
            )
            mean_shift_regress_list.append(mean_shift_regress)
            std_scale_regress_list.append(std_scale_regress)
            diff_regress_list.append(error_regress)
        if mixed:
            ### get the error with the mixed method
            mean_shift_mixed, std_scale_mixed, error_mixed = (
                difference_binomial_normal_mixed(n=n, p=p)
            )
            mean_shift_mixed_list.append(mean_shift_mixed)
            std_scale_mixed_list.append(std_scale_mixed)
            diff_mixed_list.append(error_mixed)

    ### save variables
    if compare_original:
        save_variables(
            variable_list=[
                p_list,
                n_list,
                diff_original_list,
            ],
            name_list=[
                "p_list",
                "n_list",
                "diff_list",
            ],
            path=COMPARE_ORIGINAL_FOLDER,
        )
    if optimize:
        save_variables(
            variable_list=[
                p_list,
                n_list,
                mean_shift_opt_list,
                std_scale_opt_list,
                diff_opt_list,
            ],
            name_list=[
                "p_list",
                "n_list",
                "mean_shift_opt_list",
                "std_scale_opt_list",
                "diff_opt_list",
            ],
            path=OPTIMIZE_FOLDER,
        )
    if regress:
        save_variables(
            variable_list=[
                p_list,
                n_list,
                mean_shift_regress_list,
                std_scale_regress_list,
                diff_regress_list,
            ],
            name_list=[
                "p_list",
                "n_list",
                "mean_shift_regress_list",
                "std_scale_regress_list",
                "diff_regress_list",
            ],
            path=REGRESS_FOLDER,
        )
    if mixed:
        save_variables(
            variable_list=[
                p_list,
                n_list,
                mean_shift_mixed_list,
                std_scale_mixed_list,
                diff_mixed_list,
            ],
            name_list=[
                "p_list",
                "n_list",
                "mean_shift_mixed_list",
                "std_scale_mixed_list",
                "diff_mixed_list",
            ],
            path=MIXED_FOLDER,
        )


def get_regression_parameters():
    """
    Get the regression parameters for the mean shift and std scale. Save the parameters
    to the REGRESS_FOLDER.

    Returns:
        popt_mean_shift (array):
            The optimized parameters for the mean shift regression.
        popt_std_scale (array):
            The optimized parameters for the std scale regression.
    """
    args_list = [[f"{parallel_id}"] for parallel_id in range(N_RUNS_REGRESS)]
    run_script_parallel(
        script_path="test2_deap_opt_regress.py",
        n_jobs=N_JOBS,
        args_list=args_list,
    )
    ### get best parameters for regression of mean_shift and std_scale
    best_fitness_mean_shift = 1e6
    best_fitness_std_scale = 1e6
    best_parallel_id_mean_shift = 0
    best_parallel_id_std_scale = 0
    for parallel_id in range(N_RUNS_REGRESS):
        loaded_variables = load_variables(
            name_list=[
                f"best_fitness_mean_shift_{parallel_id}",
                f"best_fitness_std_scale_{parallel_id}",
            ],
            path=REGRESS_FOLDER,
        )
        if (
            loaded_variables[f"best_fitness_mean_shift_{parallel_id}"]
            < best_fitness_mean_shift
        ):
            best_fitness_mean_shift = loaded_variables[
                f"best_fitness_mean_shift_{parallel_id}"
            ]
            best_parallel_id_mean_shift = parallel_id
        if (
            loaded_variables[f"best_fitness_std_scale_{parallel_id}"]
            < best_fitness_std_scale
        ):
            best_fitness_std_scale = loaded_variables[
                f"best_fitness_std_scale_{parallel_id}"
            ]
            best_parallel_id_std_scale = parallel_id
    # load best of mean_shift
    loaded_variables = load_variables(
        name_list=[f"popt_mean_shift_{best_parallel_id_mean_shift}"],
        path=REGRESS_FOLDER,
    )
    popt_mean_shift = loaded_variables[f"popt_mean_shift_{best_parallel_id_mean_shift}"]
    # load best of std_scale
    loaded_variables = load_variables(
        name_list=[f"popt_std_scale_{best_parallel_id_std_scale}"], path=REGRESS_FOLDER
    )
    popt_std_scale = loaded_variables[f"popt_std_scale_{best_parallel_id_std_scale}"]

    print("finished regressions")
    print(f"best fitness for mean_shift: {best_fitness_mean_shift}")
    print(f"best fitness for std_scale: {best_fitness_std_scale}")

    save_variables(
        variable_list=[popt_mean_shift, popt_std_scale],
        name_list=["popt_mean_shift", "popt_std_scale"],
        path=REGRESS_FOLDER,
    )


def preprocess_for_regress(var_value, var_name):
    """
    Normalize variable values before regression.

    Args:
        var_value (float or array):
            The original value(s) of the variable.
        var_name (str):
            The name of the variable.

    Returns:
        var_value (float or array):
            The normalized value(s) of the variable ready for regression.
    """
    ### load the dicts for normalization
    loaded_variables = load_variables(
        name_list=[
            "min_dict",
            "max_dict",
        ],
        path=REGRESS_FOLDER,
    )
    min_value = loaded_variables["min_dict"][var_name]
    max_value = loaded_variables["max_dict"][var_name]

    ### do calculations
    if var_name == "mean_shift":
        ### mean shift looks like std_scale but it is negative and has some positive
        ### coordinates (I think they are not important)
        ### make mean shift look like std_scale
        ### skip mean shift and also vmin and vmax, then rescale to 0 and 1
        var_value = -var_value
        var_value = np.clip(var_value, 0, None)
        max_value = np.max(-np.array([min_value, max_value]))
        min_value = 0

    var_value = (var_value - min_value) / (max_value - min_value)
    return var_value


def post_process_for_regression(var_value, var_name):
    """
    Denormalize variable values after regression.

    Args:
        var_value (float or array):
            The normalized value(s) of the variable.
        var_name (str):
            The name of the variable.

    Returns:
        var_value (float or array):
            The original value(s) of the variable after denormalization.
    """
    ### load the dicts for normalization
    loaded_variables = load_variables(
        name_list=[
            "min_dict",
            "max_dict",
        ],
        path=REGRESS_FOLDER,
    )
    min_value = loaded_variables["min_dict"][var_name]
    max_value = loaded_variables["max_dict"][var_name]

    if var_name == "mean_shift":
        ### skip vmin and vmax of mean shift, then rescale from 0 and 1 and then skip
        ### mean shift
        max_value = np.max(-np.array([min_value, max_value]))
        min_value = 0

    ### do calculations
    var_value = var_value * (max_value - min_value) + min_value

    if var_name == "mean_shift":
        var_value = -var_value

    if var_name == "n":
        var_value = int(np.round(var_value))
    return var_value


### TODO I have the problem that for very small p the normal distribution is not a good
### approximation of the binomial distribution.
### I think one can shift the mean and scale the standard deviation depending on the p
### and n values. I will try to optimize the shift and scale for each n and p value.

### global paramters
COMPARE_ORIGINAL = False
OPTIMIZE = False
REGRESS = True
MIXED = False
PLOT_COMPARE_ORIGINAL = True
PLOT_OPTIMIZE = True
PLOT_REGRESS = True
PLOT_MIXED = False
COMPARE_ORIGINAL_FOLDER = "test2_data_compare_original"
OPTIMIZE_FOLDER = "test2_data_optimize"
REGRESS_FOLDER = "test2_data_regress"
MIXED_FOLDER = "test2_data_mixed"
PLOTS_FOLDER = "test2_plots"
S = 10000
SEED = 1234
N_VALUES = [10, 1000, 20]
P_VALUES = [0.001, 0.1, 10]
N_JOBS = 2
N_RUNS_OPT_PER_PAIR = 100 * N_JOBS
N_RUNS_REGRESS = 15
N_PARAMS_REGRESS = 16
SCALE_ERROR_FOR_REGRESSION = True
KEEP_ONLY_IMPROVEMENTS = True


if __name__ == "__main__":

    ### create the save folder(s)
    if COMPARE_ORIGINAL:
        create_data_raw_folder(
            COMPARE_ORIGINAL_FOLDER,
        )
    if OPTIMIZE:
        create_data_raw_folder(
            OPTIMIZE_FOLDER,
        )
    if REGRESS:
        create_data_raw_folder(
            REGRESS_FOLDER,
        )
    if MIXED:
        create_data_raw_folder(
            MIXED_FOLDER,
        )

    ### compare with and without optimization
    compare_normal_binomial(
        compare_original=COMPARE_ORIGINAL, optimize=OPTIMIZE, regress=False, mixed=False
    )

    ### compare with regression (must compare original and with optimization first)
    if REGRESS:
        ### prepare the pre-processing and post-processing of the data for the regression
        loaded_variables_opt = load_variables(
            name_list=[
                "p_list",
                "n_list",
                "mean_shift_opt_list",
                "std_scale_opt_list",
                "diff_opt_list",
            ],
            path=OPTIMIZE_FOLDER,
        )
        loaded_variables = load_variables(
            name_list=[
                "diff_list",
            ],
            path=COMPARE_ORIGINAL_FOLDER,
        )
        p_arr = np.array(loaded_variables_opt["p_list"])
        n_arr = np.array(loaded_variables_opt["n_list"])
        mean_shift_opt_arr = np.array(loaded_variables_opt["mean_shift_opt_list"])
        std_scale_opt_arr = np.array(loaded_variables_opt["std_scale_opt_list"])
        diff_opt_arr = np.array(loaded_variables_opt["diff_opt_list"])
        diff_arr = np.array(loaded_variables["diff_list"])

        if SCALE_ERROR_FOR_REGRESSION:
            ### get array to weight the error for the regression depending on the
            ### improvement (reduction) of the difference by the optimization
            weight_error_arr = -np.clip(diff_opt_arr - diff_arr, None, 0)
            weight_error_arr = (weight_error_arr / np.max(weight_error_arr)) + 1.0
        else:
            weight_error_arr = np.ones_like(diff_opt_arr)

        if KEEP_ONLY_IMPROVEMENTS:
            ### keep only the improvements
            mean_shift_opt_arr[diff_opt_arr >= diff_arr] = 0
            std_scale_opt_arr[diff_opt_arr >= diff_arr] = 1

        ### create variables which can be used for pre-processing and
        ### post-processing for the regression
        min_dict = {}
        max_dict = {}
        min_dict["n"] = np.min(n_arr)
        min_dict["p"] = np.min(p_arr)
        max_dict["n"] = np.max(n_arr)
        max_dict["p"] = np.max(p_arr)
        min_dict["mean_shift"] = np.min(mean_shift_opt_arr)
        min_dict["std_scale"] = np.min(std_scale_opt_arr)
        max_dict["mean_shift"] = np.max(mean_shift_opt_arr)
        max_dict["std_scale"] = np.max(std_scale_opt_arr)

        ### save the variables prepared for the regression
        save_variables(
            variable_list=[
                min_dict,
                max_dict,
                weight_error_arr,
                mean_shift_opt_arr,
                std_scale_opt_arr,
            ],
            name_list=[
                "min_dict",
                "max_dict",
                "weight_error_arr",
                "mean_shift_opt_arr_for_regress",
                "std_scale_opt_arr_for_regress",
            ],
            path=REGRESS_FOLDER,
        )

        ### do the regression
        get_regression_parameters()
        compare_normal_binomial(
            compare_original=False, optimize=False, regress=True, mixed=False
        )

    if MIXED:
        compare_normal_binomial(
            compare_original=False, optimize=False, regress=False, mixed=True
        )

    ### plot the results
    if PLOT_COMPARE_ORIGINAL:
        plot_compare_original()

    if PLOT_OPTIMIZE:
        plot_with_transformation(mode="opt")

    if PLOT_REGRESS:
        plot_with_transformation(mode="regress")

    if PLOT_MIXED:
        plot_with_transformation(mode="mixed")
