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


def mean_shift_regression(n, p):
    x0 = n
    x1 = p

    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14 = (
        1.176321040012159,
        -6.429595249324671,
        -6.804798904871692,
        21.915210556787986,
        61.64443550309026,
        27.70993009301549,
        -27.95654152965883,
        44.87058003864243,
        -63.817886336670654,
        -40.65337691430986,
        12.246014608429185,
        -101.39842049962134,
        55.18658444426345,
        3.6975636800782237,
        19.531732368721627,
    )

    x0_min = 10
    x0_max = 1000
    x0_norm = (x0 - x0_min) / (x0_max - x0_min)
    x1_min = 0.001
    x1_max = 0.1
    x1_norm = (x1 - x1_min) / (x1_max - x1_min)
    z_min = -0.34886991656269
    z_max = 0.03020699313695153

    z_norm = np.clip(
        (
            p0
            + p1 * x0_norm
            + p2 * x1_norm
            + p3 * x0_norm**2
            + p4 * x0_norm * x1_norm
            + p5 * x1_norm**2
            + p6 * x0_norm**3
            + p7 * x0_norm**2 * x1_norm
            + p8 * x0_norm * x1_norm**2
            + p9 * x1_norm**3
            + p10 * x0_norm**4
            + p11 * x0_norm**3 * x1_norm
            + p12 * x0_norm**2 * x1_norm**2
            + p13 * x0_norm * x1_norm**3
            + p14 * x1_norm**4
        )
        ** 3,
        0,
        1,
    )
    return z_norm * (z_max - z_min) + z_min


def std_scale_regression(n, p):
    x0 = n
    x1 = p

    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14 = (
        0.320502224444166,
        6.699870528297452,
        6.935907486422781,
        -22.52956657500412,
        108.421673456727,
        -30.088834608939973,
        29.237548128991747,
        -1172.785284855411,
        -1027.3975831574996,
        58.67858836080429,
        -14.043681478309374,
        1174.4693590015047,
        -3413.5816408185233,
        1307.2105954815152,
        -44.03158341787127,
    )

    x0_min = 10
    x0_max = 1000
    x0_norm = (x0 - x0_min) / (x0_max - x0_min)
    x1_min = 0.001
    x1_max = 0.1
    x1_norm = (x1 - x1_min) / (x1_max - x1_min)
    z_min = 0.9710804893692282
    z_max = 1.6265889931274558

    z_norm = np.clip(
        (
            p0
            + p1 * x0_norm
            + p2 * x1_norm
            + p3 * x0_norm**2
            + p4 * x0_norm * x1_norm
            + p5 * x1_norm**2
            + p6 * x0_norm**3
            + p7 * x0_norm**2 * x1_norm
            + p8 * x0_norm * x1_norm**2
            + p9 * x1_norm**3
            + p10 * x0_norm**4
            + p11 * x0_norm**3 * x1_norm
            + p12 * x0_norm**2 * x1_norm**2
            + p13 * x0_norm * x1_norm**3
            + p14 * x1_norm**4
        )
        ** 3,
        0,
        1,
    )
    return z_norm * (z_max - z_min) + z_min


def gauss_1d(x, amp, mean, sig):
    return amp * np.exp(-((x - mean) ** 2) / (2 * sig**2))


def log_normal_1d(x, amp, mean, sig):
    return (amp / x) * np.exp(-((np.log(x) - mean) ** 2) / (2 * sig**2))


deap_opt_regress_path = "test2_deap_opt_regress/"


def curve_fit_func(
    X,
    g0,
    g1,
    g2,
    g3,
    g4,
    g5,
    g6,
    g7,
    g8,
    g9,
    g10,
    g11,
    g12,
    g13,
    g14,
    p0,
    # p1,
    # p2,
    # p3,
    # p4,
    # p5,
    # p6,
    # p7,
    # p8,
    # p9,
    # p10,
    # p11,
    # p12,
    # p13,
    # p14,
    # p15,
    # p16,
    # p17,
    # p18,
    # p19,
    # p20,
):
    x0, x1 = X

    ### 2D polynomial with certain degree
    return np.clip(
        p0
        + gauss_1d(
            (
                x0
                # p0
                # + p1 * x0
                # + p2 * x1
                # + p3 * x0**2
                # + p4 * x0 * x1
                # + p5 * x1**2
                # + p6 * x0**3
                # + p7 * x0**2 * x1
                # + p8 * x0 * x1**2
                # + p9 * x1**3
                # + p10 * x0**4
                # + p11 * x0**3 * x1
                # + p12 * x0**2 * x1**2
                # + p13 * x0 * x1**3
                # + p14 * x1**4
                # + p15 * x0**5
                # + p16 * x0**4 * x1
                # + p17 * x0**3 * x1**2
                # + p18 * x0**2 * x1**3
                # + p19 * x0 * x1**4
                # + p20 * x1**5
            ),
            amp=g0,
            mean=g1,
            sig=g2,
        )
        + gauss_1d(x1, amp=g3, mean=g4, sig=g5)
        + gauss_1d(x0 * x1, amp=g6, mean=g7, sig=g8)
        + gauss_1d(x0**2 * x1, amp=g9, mean=g10, sig=g11)
        + gauss_1d(x0 * x1**2, amp=g12, mean=g13, sig=g14),
        -1e20,
        1e20,
    )


def plot_2d_curve_fit_regression(
    x, y, z, sample_weight=None, vmin=None, vmax=None, grid_size=100
):
    """
    Plots a 2D color-coded image of the data with curve_fit regression and prints the regression equation.

    Parameters:
    - x: list or array of x coordinates
    - y: list or array of y coordinates
    - z: list or array of z values corresponding to the (x, y) coordinates
    - grid_size: size of the grid for plotting (default: 100)
    """
    # Check if sample_weight is provided and does not contain zeros
    if sample_weight is not None and 0 in sample_weight:
        raise ValueError("Sample weight cannot contain zeros.")

    # Normalize x, y, and z and keep the transformation for later
    x_max = np.max(x)
    x_min = np.min(x)
    y_max = np.max(y)
    y_min = np.min(y)
    z_max = np.max(z)
    z_min = np.min(z)
    x = (x - x_min) / (x_max - x_min)
    y = (y - y_min) / (y_max - y_min)
    z = (z - z_min) / (z_max - z_min)

    # Fit the curve_fit regression model
    ### do opt with deap cma in other script
    save_variables(
        name_list=["x", "y", "z"], variable_list=[x, y, z], path=deap_opt_regress_path
    )
    n_jobs = 15
    n_runs = 100 * n_jobs
    args_list = [[f"{parallel_id}"] for parallel_id in range(n_runs)]
    run_script_parallel(
        script_path="test2_deap_opt_regress.py",
        n_jobs=n_jobs,
        args_list=args_list,
    )
    ### get best parameters
    best_fitness = 1e6
    best_parallel_id = 0
    for parallel_id in range(n_runs):
        loaded_variables = load_variables(
            name_list=[f"best_fitness_{parallel_id}"], path=deap_opt_regress_path
        )
        if loaded_variables[f"best_fitness_{parallel_id}"] < best_fitness:
            best_fitness = loaded_variables[f"best_fitness_{parallel_id}"]
            best_parallel_id = parallel_id
    loaded_variables = load_variables(
        name_list=[f"popt_{best_parallel_id}"], path=deap_opt_regress_path
    )
    popt = loaded_variables[f"popt_{best_parallel_id}"]

    # Create grid for plotting
    xi = np.linspace(min(x), max(x), grid_size)
    yi = np.linspace(min(y), max(y), grid_size)
    xi, yi = np.meshgrid(xi, yi)
    zi = curve_fit_func((xi, yi), *popt)

    # Unnormalize the data
    xi = xi * (x_max - x_min) + x_min
    yi = yi * (y_max - y_min) + y_min
    zi = zi * (z_max - z_min) + z_min
    x = x * (x_max - x_min) + x_min
    y = y * (y_max - y_min) + y_min
    z = z * (z_max - z_min) + z_min

    # Plot the regression surface
    if vmin is None:
        vmin = np.min(z)
    if vmax is None:
        vmax = np.max(z)
    plt.contourf(xi, yi, zi, levels=100, cmap="viridis", vmin=vmin, vmax=vmax)
    plt.scatter(
        x,
        y,
        c=z,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        edgecolor="k",
        marker="o",
        s=(
            40 * np.array(sample_weight) / np.max(sample_weight)
            if sample_weight
            else None
        ),
    )

    ### print the regression equation and data normalization
    print(f"best_fitness: {best_fitness}")
    print(f"popt: {popt}")
    print(f"x_max: {x_max}, x_min: {x_min}")
    print(f"y_max: {y_max}, y_min: {y_min}")
    print(f"z_max: {z_max}, z_min: {z_min}")


def plot_2d_regression_image(
    x, y, z, sample_weight=None, vmin=None, vmax=None, degree=2, grid_size=100
):
    """
    Plots a 2D color-coded image of the data with polynomial regression and plots the regression equation.

    Parameters:
    - x: list or array of x coordinates
    - y: list or array of y coordinates
    - z: list or array of z values corresponding to the (x, y) coordinates
    - degree: degree of the polynomial regression (default: 2)
    - grid_size: size of the grid for plotting (default: 100)
    """
    # Normalize x and y and keep the transformation for later
    x_mean = np.mean(x)
    x_std = np.std(x)
    x_max = np.max(x)
    x_min = np.min(x)
    y_mean = np.mean(y)
    y_std = np.std(y)
    y_max = np.max(y)
    y_min = np.min(y)
    x = (x - x_min) / (x_max - x_min)
    y = (y - y_min) / (y_max - y_min)

    # Prepare the data for polynomial regression
    X = np.column_stack((x, y))
    # Create a polynomial regression pipeline
    polynomial_model = make_pipeline(PolynomialFeatures(degree), LinearRegression())

    # Fit the model
    polynomial_model.fit(
        X,
        z,
        linearregression__sample_weight=(
            sample_weight if sample_weight is not None else None
        ),
    )

    # Predict new values for the surface plot
    xi = np.linspace(min(x), max(x), grid_size)
    yi = np.linspace(min(y), max(y), grid_size)
    xi, yi = np.meshgrid(xi, yi)
    Xi = np.column_stack((xi.ravel(), yi.ravel()))
    zi = polynomial_model.predict(Xi)
    zi = zi.reshape(xi.shape)

    # Unnormalize the x,y values
    xi = xi * (x_max - x_min) + x_min
    yi = yi * (y_max - y_min) + y_min
    x = x * (x_max - x_min) + x_min
    y = y * (y_max - y_min) + y_min

    # Plot the regression surface
    if vmin is None:
        vmin = np.min(z)
    if vmax is None:
        vmax = np.max(z)
    plt.contourf(xi, yi, zi, levels=100, cmap="viridis", vmin=vmin, vmax=vmax)

    # Plot the original data points, scaled by the sample weight
    plt.scatter(
        x,
        y,
        c=z,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        edgecolor="k",
        marker="o",
        s=(
            40 * np.array(sample_weight) / np.max(sample_weight)
            if sample_weight
            else None
        ),
    )

    # Print the regression equation
    # TODO


def plot_2d_interpolated_image(
    x, y, z, vmin=None, vmax=None, grid_size=100, method="linear"
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

    print(f"max interpolated: {np.max(zi)}")

    # Plot the interpolated data
    if vmin is None:
        vmin = np.min(z)
    if vmax is None:
        vmax = np.max(z)
    plt.contourf(xi, yi, zi, levels=100, cmap="viridis", vmin=vmin, vmax=vmax)

    # plot scatter plot of original data
    plt.scatter(
        x, y, c=z, cmap="viridis", vmin=vmin, vmax=vmax, edgecolor="k", marker="o"
    )


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
    print(f"Optimize for n={n}, p={p}")
    ### save p and n to be availyble in optimization script
    save_variables(
        variable_list=[p, n],
        name_list=["p", "n"],
        path=OPTIMIZE_FOLDER,
    )
    ### run optimization
    args_list = [[f"{parallel_id}"] for parallel_id in range(N_RUNS)]
    run_script_parallel(
        script_path="test2_deap_opt_transform.py",
        n_jobs=N_JOBS,
        args_list=args_list,
    )
    ### get best mean_shift and std_scale by loading all optimizations and check best
    ### fitness
    best_fitness = 1e6
    best_parallel_id = 0
    for parallel_id in range(N_RUNS):
        loaded_variables = load_variables(
            name_list=[
                f"error_opt_{parallel_id}",
            ],
            path=OPTIMIZE_FOLDER,
        )
        error_opt = loaded_variables[f"error_opt_{parallel_id}"]
        print(f"n={n}, p={p}, error_opt_{parallel_id}: {error_opt}")
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


def plot_optimize():
    """
    Plot the difference between the binomial and normal distribution for various n and
    p values. Further plots the optimized mean_shift and std_scale values. Load the data
    from the OPTIMIZE_FOLDER and save the plot to the PLOTS_FOLDER.
    """
    ### load the data
    loaded_variables = load_variables(
        name_list=[
            "p_list",
            "n_list",
            "mean_shift_opt_list",
            "std_scale_opt_list",
            "diff_opt_list",
        ],
        path=OPTIMIZE_FOLDER,
    )
    ### plot the data
    plt.figure(figsize=(6.4 * 2, 4.8 * 2 * 3))
    for idx, title, key in [
        (1, "Mean Shift optimized", "mean_shift_opt_list"),
        (2, "Std Scale optimized", "std_scale_opt_list"),
        (3, "Difference optimized", "diff_opt_list"),
    ]:
        plt.subplot(3, 1, idx)
        plot_2d_interpolated_image(
            x=loaded_variables["n_list"],
            y=loaded_variables["p_list"],
            z=loaded_variables[key],
            vmin=0,
            vmax=np.max(loaded_variables[key]),
        )
        plt.colorbar()
        plt.xlabel("n")
        plt.ylabel("p")
        plt.title(f"{title}\n(max: {np.max(loaded_variables[key])})")
    plt.tight_layout()
    create_dir(PLOTS_FOLDER)
    plt.savefig(f"{PLOTS_FOLDER}/difference_optimized.png", dpi=300)


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
    )
    plt.colorbar()
    plt.xlabel("n")
    plt.ylabel("p")
    plt.title(f"Difference original\n(max: {np.max(loaded_variables['diff_list'])})")
    plt.tight_layout()
    create_dir(PLOTS_FOLDER)
    plt.savefig(f"{PLOTS_FOLDER}/difference_original.png", dpi=300)


def compare_with_or_without_optimization():
    """
    Compare the difference between the binomial and normal distribution for various n and
    p values with and without optimization. Save the data to the COMPARE_ORIGINAL_FOLDER
    and OPTIMIZE_FOLDER.
    """
    ### create the save folder(s)
    if COMPARE_ORIGINAL:
        create_data_raw_folder(
            COMPARE_ORIGINAL_FOLDER,
        )
    if OPTIMIZE:
        create_data_raw_folder(
            OPTIMIZE_FOLDER,
        )

    ### create the n/p pairs
    n_arr = logarithmic_arange(*N_VALUES).astype(int)
    p_arr = logarithmic_arange(*P_VALUES)

    ### get difference between binomial and normal distribution for each n/p pair
    p_list = []
    n_list = []
    diff_original_list = []
    mean_shift_opt_list = []
    std_scale_opt_list = []
    diff_opt_list = []
    for p in p_arr:
        for n in n_arr:
            p_list.append(p)
            n_list.append(n)
            if COMPARE_ORIGINAL:
                ### get the error without optimization
                error = difference_binomial_normal(mean_shift=0, std_scale=1, n=n, p=p)
                diff_original_list.append(error)
            if OPTIMIZE:
                ### get the error with optimization
                mean_shift_opt, std_scale_opt, error_opt = (
                    difference_binomial_normal_optimize(n=n, p=p)
                )
                mean_shift_opt_list.append(mean_shift_opt)
                std_scale_opt_list.append(std_scale_opt)
                diff_opt_list.append(error_opt)

    ### save variables
    if COMPARE_ORIGINAL:
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
    if OPTIMIZE:
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


### TODO I have the problem that for very small p the normal distribution is not a good
### approximation of the binomial distribution.
### I think one can shift the mean and scale the standard deviation depending on the p
### and n values. I will try to optimize the shift and scale for each n and p value.

### global paramters
COMPARE_ORIGINAL = True
OPTIMIZE = True
REGRESS = True
PLOT_COMPARE_ORIGINAL = True
PLOT_OPTIMIZE = True
PLOT_REGRESS = True
COMPARE_ORIGINAL_FOLDER = "test2_data_compare_original"
OPTIMIZE_FOLDER = "test2_data_optimize"
PLOTS_FOLDER = "test2_plots"
S = 10000
SEED = 1234
N_VALUES = [10, 1000, 2]  # 20]
P_VALUES = [0.001, 0.1, 2]  # 10]
N_JOBS = 2
N_RUNS = 100 * N_JOBS


if __name__ == "__main__":

    ### TODO: restructure this thing

    # 1st compare binomial and normal samples for various n and p values, save: p_list, n_list and diff_list
    # 2nd optimize mean shift and std scale for each n and p value and get improved error, save: mean_shift_list, std_scale_list and error_improved_list
    # 3rd make a 2D regression for the optimized mean shift and std scale, get mean_shift_regress(n, p) and std_scale_regress(n, p), save: the optimized parameters of the regression equations
    # 4th plot: (1) error depending on n and p, (2) optimized mean shift and std scale depending on n and p and corresponding error improvement, (3) regressed mean shift and std scale depending on n and p and corresponding error improvement

    if COMPARE_ORIGINAL or OPTIMIZE:
        compare_with_or_without_optimization()

    if PLOT_COMPARE_ORIGINAL:
        plot_compare_original()

    if PLOT_OPTIMIZE:
        plot_optimize()

    quit()

    OPTIMIZE = False
    PLOT_OPTIMIZED = True
    USE_REGRESSION = False
    PLOT_REGRESSION = False

    ### 1st optimize mean shift and std scale for each n and p value
    n_arr = logarithmic_arange(10, 1000, 20).astype(int)
    p_arr = logarithmic_arange(0.001, 0.1, 10)

    ### 1st get errors for all n and p values without optimization
    p_list = []
    n_list = []
    error_list = []
    for p in p_arr:
        for n in n_arr:
            ### get the error without optimization
            error = objective_function(mean_shift=0, std_scale=1, n=n, p=p, m=M)
            error_list.append(error)

            ### store the results
            p_list.append(p)
            n_list.append(n)

    ### save variables
    save_variables(
        variable_list=[
            p_list,
            n_list,
            error_list,
        ],
        name_list=[
            "p_list",
            "n_list",
            "error_list",
        ],
        path="data_optimize_binomial_normal",
    )

    ### plot the original error
    # original error -> interpolation plot
    plt.figure(figsize=(6.4 * 2, 4.8 * 2))
    plt.subplot(1, 1, 1)
    plot_2d_interpolated_image(
        x=n_list, y=p_list, z=error_list, vmin=0, vmax=np.max(error_list)
    )
    plt.colorbar()
    plt.xlabel("n")
    plt.ylabel("p")
    plt.title(f"Error original\n(max: {np.max(error_list)})")
    plt.tight_layout()
    plt.savefig("test2_01_error_original.png", dpi=300)

    ### 2nd optimize mean shift and std scale for each n and p value and get improved error
    if OPTIMIZE:
        loaded_variables = load_variables(
            name_list=[
                "p_list",
                "n_list",
                "error_list",
            ],
            path="data_optimize_binomial_normal",
        )
        p_list = loaded_variables["p_list"]
        n_list = loaded_variables["n_list"]
        error_list = loaded_variables["error_list"]
        mean_shift_list = []
        std_scale_list = []
        error_improved_list = []
        for p, n in zip(p_list, n_list):
            ### save p and n to be availyble in optimization script
            save_variables(
                variable_list=[p, n],
                name_list=["p", "n"],
                path=OPTIMIZE_FOLDER,
            )
            ### run optimization
            n_jobs = 15
            n_runs = 100 * n_jobs
            args_list = [[f"{parallel_id}"] for parallel_id in range(n_runs)]
            run_script_parallel(
                script_path="test2_deap_opt_transform.py",
                n_jobs=n_jobs,
                args_list=args_list,
            )
            ### get best parameters
            best_fitness = 1e6
            best_parallel_id = 0
            for parallel_id in range(n_runs):
                loaded_variables = load_variables(
                    name_list=[
                        f"error_improved_{parallel_id}",
                    ],
                    path=OPTIMIZE_FOLDER,
                )
                error_improved = loaded_variables[f"error_improved_{parallel_id}"]

                if error_improved < best_fitness:
                    best_fitness = error_improved
                    best_parallel_id = parallel_id
            loaded_variables = load_variables(
                name_list=[
                    f"mean_shift_{best_parallel_id}",
                    f"std_scale_{best_parallel_id}",
                    f"error_improved_{best_parallel_id}",
                ],
                path=OPTIMIZE_FOLDER,
            )
            mean_shift = loaded_variables[f"mean_shift_{best_parallel_id}"]
            std_scale = loaded_variables[f"std_scale_{best_parallel_id}"]
            error_improved = loaded_variables[f"error_improved_{best_parallel_id}"]

            ### store the results
            error_improved_list.append(error_improved)
            mean_shift_list.append(mean_shift)
            std_scale_list.append(std_scale)

        ### save variables
        save_variables(
            variable_list=[
                mean_shift_list,
                std_scale_list,
                error_improved_list,
            ],
            name_list=[
                "mean_shift_list",
                "std_scale_list",
                "error_improved_list",
            ],
            path="data_optimize_binomial_normal",
        )

    ### 4th plot the optimized error with optimized mean shift and std scale
    ### also calculate the ŕegression for the optimized mean shift and std scale
    if PLOT_OPTIMIZED:
        # fitting improved error -> interpolation plot
        # fitting improvement -> interpolation plot
        # fitting mean shift -> regression plot
        # fitting std scale -> regression plot
        loaded_variables = load_variables(
            name_list=[
                "error_improved_list",
                "mean_shift_list",
                "std_scale_list",
            ],
            path="data_optimize_binomial_normal",
        )
        error_improved_list = loaded_variables["error_improved_list"]
        mean_shift_list = loaded_variables["mean_shift_list"]
        std_scale_list = loaded_variables["std_scale_list"]
        # ### TODO tmp
        # error_improved_list = np.random.rand(len(error_improved_list))
        # mean_shift_list = np.random.rand(len(mean_shift_list))
        # std_scale_list = np.random.rand(len(std_scale_list))
        # ### TODO tmp
        error_change_arr = np.array(error_improved_list) - np.array(error_list)
        improvement_arr = -np.clip(error_change_arr, None, 0)
        improvement_arr_norm = improvement_arr / np.max(improvement_arr)

        ### scale the mean shift and std scale by the error improvement
        ### --> only keep the transformations which improve the error
        alpha = improvement_arr_norm
        mean_shift_list = alpha * np.array(mean_shift_list) + (1 - alpha) * 0
        std_scale_list = alpha * np.array(std_scale_list) + (1 - alpha) * 1

        ### the mean shift is mostly 0 and at some positions negative, multiply it by -1
        mean_shift_list = -mean_shift_list

        plt.figure(figsize=(6.4 * 2 * 2, 4.8 * 2 * 4))
        plt.subplot(4, 2, 1)
        plot_2d_interpolated_image(
            x=n_list,
            y=p_list,
            z=error_improved_list,
            vmin=0,
            vmax=np.max(error_improved_list),
        )
        plt.colorbar()
        plt.xlabel("n")
        plt.ylabel("p")
        plt.title(f"Error optimized\n(max: {np.max(error_improved_list)})")
        plt.subplot(4, 2, 3)
        plot_2d_interpolated_image(
            x=n_list,
            y=p_list,
            z=error_change_arr,
            vmin=-np.max(np.abs(error_change_arr)),
            vmax=np.max(np.abs(error_change_arr)),
        )
        plt.colorbar()
        plt.xlabel("n")
        plt.ylabel("p")
        plt.title("Error improvement")
        plt.subplot(4, 2, 5)
        plot_2d_interpolated_image(
            x=n_list,
            y=p_list,
            z=mean_shift_list,
            vmin=-np.max(np.abs(mean_shift_list)),
            vmax=np.max(np.abs(mean_shift_list)),
        )
        plt.colorbar()
        plt.xlabel("n")
        plt.ylabel("p")
        plt.title("Mean shift")
        plt.subplot(4, 2, 7)
        plot_2d_interpolated_image(
            x=n_list,
            y=p_list,
            z=std_scale_list,
            vmin=1 - np.max(1 - np.array(std_scale_list)),
            vmax=1 + np.max(np.array(std_scale_list) - 1),
        )
        plt.colorbar()
        plt.xlabel("n")
        plt.ylabel("p")
        plt.title("Standard deviation scale")
        plt.subplot(4, 2, 6)
        plot_2d_curve_fit_regression(
            x=n_list,
            y=p_list,
            z=mean_shift_list,
            vmin=-np.max(np.abs(mean_shift_list)),
            vmax=np.max(np.abs(mean_shift_list)),
            # sample_weight=-np.clip(error_change_arr, None, 0)
            # + 0.01 * np.max(improvement_arr),
            # degree=3,
        )
        plt.colorbar()
        plt.xlabel("n")
        plt.ylabel("p")
        plt.title("Mean shift regression")
        plt.subplot(4, 2, 8)
        plot_2d_curve_fit_regression(
            x=n_list,
            y=p_list,
            z=std_scale_list,
            vmin=1 - np.max(1 - np.array(std_scale_list)),
            vmax=1 + np.max(np.array(std_scale_list) - 1),
            # sample_weight=-np.clip(error_change_arr, None, 0)
            # + 0.01 * np.max(improvement_arr),
            # degree=3,
        )
        plt.colorbar()
        plt.xlabel("n")
        plt.ylabel("p")
        plt.title("Standard deviation scale regression")
        plt.tight_layout()
        plt.savefig("test2_02_error_optimized.png", dpi=300)

    ### 3rd use regression for mean shift and std scale and recalculate the improved error
    if USE_REGRESSION:
        ### load the optimized parameters and corresponding original and optimized errors
        loaded_variables = load_variables(
            name_list=[
                "p_list",
                "n_list",
            ],
            path="data_optimize_binomial_normal",
        )
        p_list = loaded_variables["p_list"]
        n_list = loaded_variables["n_list"]

        ### use regression equations to recalculate mean shift and std scale
        mean_shift_reg_list = []
        std_scale_reg_list = []
        error_improved_reg_list = []
        for p, n in zip(p_list, n_list):
            ### get the optimized parameters and best error
            mean_shift = mean_shift_regression(n, p)
            std_scale = std_scale_regression(n, p)
            error_improved = objective_function(
                mean_shift=mean_shift, std_scale=std_scale, n=n, p=p, m=M
            )

            ### store the results
            error_improved_reg_list.append(error_improved)
            mean_shift_reg_list.append(mean_shift)
            std_scale_reg_list.append(std_scale)

        ### save variables
        save_variables(
            variable_list=[
                mean_shift_reg_list,
                std_scale_reg_list,
                error_improved_reg_list,
            ],
            name_list=[
                "mean_shift_reg_list",
                "std_scale_reg_list",
                "error_improved_reg_list",
            ],
            path="data_optimize_binomial_normal",
        )

    ### 5th plot the regression error with regressed mean shift and std scale and compare it
    ### with the optimized error
    if PLOT_REGRESSION:
        # regression improved error -> interpolation plot
        # regression improvement -> interpolation plot
        # regression mean shift -> regression plot
        # regression std scale -> regression plot
        loaded_variables = load_variables(
            name_list=[
                "error_improved_list",
                "mean_shift_list",
                "std_scale_list",
                "error_improved_reg_list",
                "mean_shift_reg_list",
                "std_scale_reg_list",
            ],
            path="data_optimize_binomial_normal",
        )
        error_improved_list = loaded_variables["error_improved_list"]
        mean_shift_list = loaded_variables["mean_shift_list"]
        std_scale_list = loaded_variables["std_scale_list"]
        error_improved_reg_list = loaded_variables["error_improved_reg_list"]
        mean_shift_reg_list = loaded_variables["mean_shift_reg_list"]
        std_scale_reg_list = loaded_variables["std_scale_reg_list"]

        plt.figure(figsize=(6.4 * 2, 4.8 * 2 * 4))
        plt.subplot(4, 1, 1)
        plot_2d_interpolated_image(
            x=n_list,
            y=p_list,
            z=error_improved_reg_list,
            vmin=0,
            vmax=np.max(error_improved_reg_list),
        )
        plt.colorbar()
        plt.xlabel("n")
        plt.ylabel("p")
        plt.title(f"Error optimized\n(max: {np.max(error_improved_reg_list)})")
        plt.subplot(4, 1, 2)
        plot_2d_interpolated_image(
            x=n_list,
            y=p_list,
            z=np.array(error_improved_reg_list) - np.array(error_list),
            vmin=-np.max(
                np.abs(np.array(error_improved_reg_list) - np.array(error_list))
            ),
            vmax=np.max(
                np.abs(np.array(error_improved_reg_list) - np.array(error_list))
            ),
        )
        plt.colorbar()
        plt.xlabel("n")
        plt.ylabel("p")
        plt.title("Error improvement")
        plt.subplot(4, 1, 3)
        plot_2d_interpolated_image(
            x=n_list,
            y=p_list,
            z=mean_shift_reg_list,
            vmin=-np.max(np.abs(mean_shift_reg_list)),
            vmax=np.max(np.abs(mean_shift_reg_list)),
        )
        plt.colorbar()
        plt.xlabel("n")
        plt.ylabel("p")
        plt.title("Mean shift")
        plt.subplot(4, 1, 4)
        plot_2d_interpolated_image(
            x=n_list,
            y=p_list,
            z=std_scale_reg_list,
            vmin=1 - np.max(1 - np.array(std_scale_reg_list)),
            vmax=1 + np.max(np.array(std_scale_reg_list) - 1),
        )
        plt.colorbar()
        plt.xlabel("n")
        plt.ylabel("p")
        plt.title("Standard deviation scale")
        plt.tight_layout()
        plt.savefig("test2_03_error_regression.png", dpi=300)

        # difference fitting/regression improved error -> interpolation plot
        # difference fitting/regression improvement -> interpolation plot
        # difference fitting/regression mean shift -> interpolation plot
        # difference fitting/regression std scale -> interpolation plot
        plt.figure(figsize=(6.4 * 2, 4.8 * 2 * 4))
        plt.subplot(4, 1, 1)
        plot_2d_interpolated_image(
            x=n_list,
            y=p_list,
            z=np.array(error_improved_list) - np.array(error_improved_reg_list),
            vmin=-np.max(
                np.abs(
                    np.array(error_improved_list) - np.array(error_improved_reg_list)
                )
            ),
            vmax=np.max(
                np.abs(
                    np.array(error_improved_list) - np.array(error_improved_reg_list)
                )
            ),
        )
        plt.colorbar()
        plt.xlabel("n")
        plt.ylabel("p")
        plt.title("Error difference between optimized and regression")
        plt.subplot(4, 1, 2)
        plot_2d_interpolated_image(
            x=n_list,
            y=p_list,
            z=np.array(error_improved_list)
            - np.array(error_list)
            - np.array(error_improved_reg_list)
            + np.array(error_list),
            vmin=-np.max(
                np.abs(
                    np.array(error_improved_list)
                    - np.array(error_list)
                    - np.array(error_improved_reg_list)
                    + np.array(error_list)
                )
            ),
            vmax=np.max(
                np.abs(
                    np.array(error_improved_list)
                    - np.array(error_list)
                    - np.array(error_improved_reg_list)
                    + np.array(error_list)
                )
            ),
        )
        plt.colorbar()
        plt.xlabel("n")
        plt.ylabel("p")
        plt.title("Error improvement difference between optimized and regression")
        plt.subplot(4, 1, 3)
        plot_2d_interpolated_image(
            x=n_list,
            y=p_list,
            z=np.array(mean_shift_list) - np.array(mean_shift_reg_list),
            vmin=-np.max(
                np.abs(np.array(mean_shift_list) - np.array(mean_shift_reg_list))
            ),
            vmax=np.max(
                np.abs(np.array(mean_shift_list) - np.array(mean_shift_reg_list))
            ),
        )
        plt.colorbar()
        plt.xlabel("n")
        plt.ylabel("p")
        plt.title("Mean shift difference between optimized and regression")
        plt.subplot(4, 1, 4)
        plot_2d_interpolated_image(
            x=n_list,
            y=p_list,
            z=np.array(std_scale_list) - np.array(std_scale_reg_list),
            vmin=-np.max(
                np.abs(np.array(std_scale_list) - np.array(std_scale_reg_list))
            ),
            vmax=np.max(
                np.abs(np.array(std_scale_list) - np.array(std_scale_reg_list))
            ),
        )
        plt.colorbar()
        plt.xlabel("n")
        plt.ylabel("p")
        plt.title(
            "Standard deviation scale difference between optimized and regression"
        )
        plt.tight_layout()
        plt.savefig("test2_04_error_difference.png", dpi=300)
