import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from CompNeuroPy import DeapCma, save_variables, load_variables
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
    def curve_fit_func(
        X,
        p0,
        p1,
        p2,
        p3,
        p4,
        p5,
        p6,
        p7,
        p8,
        p9,
        p10,
        p11,
        p12,
        p13,
        p14,
        p15,
        p16,
        p17,
        p18,
        p19,
        p20,
    ):
        x0, x1 = X
        ### 2D polynomial with certain degree
        return np.clip(
            (
                p0
                + p1 * x0
                + p2 * x1
                + p3 * x0**2
                + p4 * x0 * x1
                + p5 * x1**2
                + p6 * x0**3
                + p7 * x0**2 * x1
                + p8 * x0 * x1**2
                + p9 * x1**3
                + p10 * x0**4
                + p11 * x0**3 * x1
                + p12 * x0**2 * x1**2
                + p13 * x0 * x1**3
                + p14 * x1**4
                + p15 * x0**5
                + p16 * x0**4 * x1
                + p17 * x0**3 * x1**2
                + p18 * x0**2 * x1**3
                + p19 * x0 * x1**4
                + p20 * x1**5
            )
            ** 3,
            np.min(z),
            np.max(z),
        )

    def curve_fit_evaluate_function(population):
        loss_list = []
        ### the population is a list of individuals which are lists of parameters
        for individual in population:
            loss_of_individual = curve_fit_objective_function(individual)
            loss_list.append((loss_of_individual,))
        return loss_list

    def curve_fit_objective_function(individual):
        is_data = curve_fit_func((x, y), *individual)
        target_data = z
        return np.sum((is_data - target_data) ** 2)

    # ### do opt with scipy curve_fit
    # popt, pcov = curve_fit(
    #     curve_fit_func,
    #     (x, y),
    #     z,
    #     sigma=1 / sample_weight if sample_weight is not None else None,
    #     absolute_sigma=False,
    # )

    ### do opt with deap cma
    param_names = [
        "p0",
        "p1",
        "p2",
        "p3",
        "p4",
        "p5",
        "p6",
        "p7",
        "p8",
        "p9",
        "p10",
        "p11",
        "p12",
        "p13",
        "p14",
        "p15",
        "p16",
        "p17",
        "p18",
        "p19",
        "p20",
    ]
    ### run the optimization, get popt
    best_fitness = 1e9
    ### create progress bar showing the best fitness
    progress_bar = tqdm(range(10), total=10)
    for _ in progress_bar:
        deap_cma = DeapCma(
            lower=np.array([-1] * len(param_names)),
            upper=np.array([1] * len(param_names)),
            evaluate_function=curve_fit_evaluate_function,
            param_names=param_names,
            hard_bounds=False,
            display_progress_bar=False,
        )
        deap_cma_result = deap_cma.run(max_evals=2000)
        if deap_cma_result["best_fitness"] < best_fitness:
            best_fitness = deap_cma_result["best_fitness"]
            popt = [deap_cma_result[param_name] for param_name in param_names]
        progress_bar.set_description(f"Best Fitness: {best_fitness}")

    # ### solve polynomial using linalg
    # x0 = x
    # x1 = y
    # n_samples = len(x0)
    # X = np.column_stack(
    #     [
    #         np.ones(n_samples),
    #         x0,
    #         x1,
    #         x0**2,
    #         x0 * x1,
    #         x1**2,
    #         x0**3,
    #         x0**2 * x1,
    #         x0 * x1**2,
    #         x1**3,
    #         x0**4,
    #         x0**3 * x1,
    #         x0**2 * x1**2,
    #         x0 * x1**3,
    #         x1**4,
    #     ]
    # )

    # # Solve the least squares problem
    # coefficients, residuals, rank, s = np.linalg.lstsq(X, z, rcond=None)
    # popt = coefficients

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
    Plots a 2D color-coded image of the data with interpolation and extrapolation.

    Parameters:
    - x: list or array of x coordinates
    - y: list or array of y coordinates
    - z: list or array of z values corresponding to the (x, y) coordinates
    - grid_size: size of the interpolation grid (default: 100)
    - method: interpolation method, options are 'linear', 'nearest', 'cubic' (default: 'linear')
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

    # find local extrema of the surface using scipy
    from scipy.signal import argrelextrema

    # find local maxima
    maxima = argrelextrema(zi, np.greater, axis=None)


def generate_samples(n, p, m, mean_shift=0, std_scale=1):
    # Generate data samples
    binomial_sample = np.random.binomial(n, p, m)
    mean = n * p
    std_dev = np.sqrt(n * p * (1 - p))
    normal_sample = np.random.normal(mean + mean_shift, std_dev * std_scale, m)

    ### round and clip the normal sample
    normal_sample = np.round(normal_sample)
    normal_sample[normal_sample < 0] = 0
    normal_sample[normal_sample > n] = n

    return binomial_sample, normal_sample


def get_error_of_samples(binomial_sample, normal_sample, m):
    diff = (
        np.histogram(binomial_sample, bins=n + 1, range=(-0.5, n + 0.5))[0]
        - np.histogram(normal_sample, bins=n + 1, range=(-0.5, n + 0.5))[0]
    )
    return np.sum(np.abs(diff)) / (2 * m)


def objective_function(mean_shift, std_scale):
    # Generate data samples
    binomial_sample, normal_sample = generate_samples(
        n=N, p=P, m=M, mean_shift=mean_shift, std_scale=std_scale
    )

    # Calculate error
    error = get_error_of_samples(binomial_sample, normal_sample, m=M)
    return error


def objective_function_for_minimize(x):
    # print(f"P: {P}, N: {N}, mean_shift: {x[0]}, std_scale: {x[1]}")
    return objective_function(mean_shift=x[0], std_scale=x[1])


def evaluate_function(population):
    loss_list = []
    ### the population is a list of individuals which are lists of parameters
    for individual in population:
        loss_of_individual = objective_function(
            mean_shift=individual[0], std_scale=individual[1]
        )
        loss_list.append((loss_of_individual,))
    return loss_list


def logarithmic_distribution(start, end, num_points):
    """
    Generate a list of logarithmically spaced points between a start and end point.

    Parameters:
    start (float): The starting point of the distribution.
    end (float): The ending point of the distribution.
    num_points (int): The number of points to generate.

    Returns:
    list: A list of logarithmically spaced points.
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


### TODO I have the problem that for very small p the normal distribution is not a good
### approximation of the binomial distribution.
### I think one can shift the mean and scale the standard deviation depending on the p
### and n values. I will try to optimize the shift and scale for each n and p value.

OPTIMIZE = False
PLOT_OPTIMIZED = False
USE_REGRESSION = False
PLOT_REGRESSION = False

### 1st optimize mean shift and std scale for each n and p value
n_arr = logarithmic_distribution(10, 1000, 20).astype(int)
p_arr = logarithmic_distribution(0.001, 0.1, 10)

### bounds for optimized parameters
shift_mean_bounds = [-1, 1]
scale_std_bounds = [0.5, 2]
lb = np.array([shift_mean_bounds[0], scale_std_bounds[0]])
ub = np.array([shift_mean_bounds[1], scale_std_bounds[1]])
# number of samples
M = 10000

### 1st get errors for all n and p values without optimization
p_list = []
n_list = []
error_list = []
for p in p_arr:
    for n in n_arr:
        ### set the global variables probability of success and number of trials
        P = p
        N = n

        ### get the error without optimization
        error = objective_function(mean_shift=0, std_scale=1)
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
        ### set the global variables probability of success and number of trials
        P = p
        N = n

        # ### optimize the mean shift and standard deviation scale using scipy minimize
        # result = minimize(
        #     objective_function_for_minimize,
        #     x0=[0, 1],
        #     bounds=Bounds(lb, ub),
        #     method="Nelder-Mead",
        # )
        # mean_shift = result.x[0]
        # std_scale = result.x[1]
        # error_improved = result.fun

        ### create an instance of the DeapCma class
        deap_cma = DeapCma(
            lower=lb,
            upper=ub,
            evaluate_function=evaluate_function,
            param_names=["mean_shift", "std_scale"],
            hard_bounds=True,
        )

        ### run the optimization
        deap_cma_result = deap_cma.run(max_evals=1000)

        ### get the optimized parameters and best error
        mean_shift = deap_cma_result["mean_shift"]
        std_scale = deap_cma_result["std_scale"]
        error_improved = deap_cma_result["best_fitness"]

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
        ### set the global variables probability of success and number of trials
        P = p
        N = n

        ### get the optimized parameters and best error
        mean_shift = mean_shift_regression(n, p)
        std_scale = std_scale_regression(n, p)
        error_improved = objective_function(mean_shift=mean_shift, std_scale=std_scale)

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

### 4th plot the original error
# original error -> interpolation plot
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

### 5th plot the optimized error with optimized mean shift and std scale
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
    error_change_arr = np.array(error_improved_list) - np.array(error_list)
    improvement_arr = -np.clip(error_change_arr, None, 0)
    improvement_arr_norm = improvement_arr / np.max(improvement_arr)

    ### scale the mean shift and std scale by the error improvement
    ### --> only keep the transformations which improve the error
    alpha = improvement_arr_norm
    mean_shift_list = alpha * np.array(mean_shift_list) + (1 - alpha) * 0
    std_scale_list = alpha * np.array(std_scale_list) + (1 - alpha) * 1

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

### 6th plot the regression error with regressed mean shift and std scale and compare it
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
        vmin=-np.max(np.abs(np.array(error_improved_reg_list) - np.array(error_list))),
        vmax=np.max(np.abs(np.array(error_improved_reg_list) - np.array(error_list))),
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
            np.abs(np.array(error_improved_list) - np.array(error_improved_reg_list))
        ),
        vmax=np.max(
            np.abs(np.array(error_improved_list) - np.array(error_improved_reg_list))
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
        vmin=-np.max(np.abs(np.array(mean_shift_list) - np.array(mean_shift_reg_list))),
        vmax=np.max(np.abs(np.array(mean_shift_list) - np.array(mean_shift_reg_list))),
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
        vmin=-np.max(np.abs(np.array(std_scale_list) - np.array(std_scale_reg_list))),
        vmax=np.max(np.abs(np.array(std_scale_list) - np.array(std_scale_reg_list))),
    )
    plt.colorbar()
    plt.xlabel("n")
    plt.ylabel("p")
    plt.title("Standard deviation scale difference between optimized and regression")
    plt.tight_layout()
    plt.savefig("test2_04_error_difference.png", dpi=300)

quit()
# Plot the error as a function of p and n as a heatmap
plt.figure(figsize=(20, 15))
### scatter plot with max from error_list and error_improved_list
plt.subplot(2, 2, 1)

plot_2d_interpolated_image(
    x=n_list,
    y=p_list,
    z=error_list,
    vmin=0,
    vmax=np.max(error_list + error_improved_list),
)
print(np.max(error_list + error_improved_list))
plt.scatter(
    n_list,
    p_list,
    c=error_list,
    cmap="viridis",
    vmin=0,
    vmax=np.max(error_list + error_improved_list),
    edgecolor="k",
    marker="o",
)
plt.colorbar()
plt.xlabel("n")
plt.ylabel("p")
plt.title(f"Error original\n(max: {np.max(error_list)})")
plt.subplot(2, 2, 2)
plot_2d_interpolated_image(
    x=n_list,
    y=p_list,
    z=error_improved_list,
    vmin=0,
    vmax=np.max(error_list + error_improved_list),
)
plt.scatter(
    n_list,
    p_list,
    c=error_improved_list,
    cmap="viridis",
    vmin=0,
    vmax=np.max(error_list + error_improved_list),
    edgecolor="k",
    marker="o",
)
plt.colorbar()
plt.xlabel("n")
plt.ylabel("p")
plt.title(f"Error optimized\n(max: {np.max(error_improved_list)})")
plt.subplot(2, 2, 3)
plot_2d_regression_image(
    x=n_list,
    y=p_list,
    z=mean_shift_list,
    sample_weight=error_list,
    vmin=-np.max(np.abs(mean_shift_list)),
    vmax=np.max(np.abs(mean_shift_list)),
    degree=4,
)
plt.scatter(
    n_list,
    p_list,
    c=mean_shift_list,
    cmap="viridis",
    vmin=-np.max(np.abs(mean_shift_list)),
    vmax=np.max(np.abs(mean_shift_list)),
    edgecolor="k",
    marker="o",
)
plt.colorbar()
plt.xlabel("n")
plt.ylabel("p")
plt.title("Mean shift")
plt.subplot(2, 2, 4)
plot_2d_regression_image(
    x=n_list,
    y=p_list,
    z=std_scale_list,
    sample_weight=error_list,
    vmin=1 - np.max(1 - np.array(std_scale_list)),
    vmax=1 + np.max(np.array(std_scale_list) - 1),
    degree=4,
)
plt.scatter(
    n_list,
    p_list,
    c=std_scale_list,
    cmap="viridis",
    vmin=1 - np.max(1 - np.array(std_scale_list)),
    vmax=1 + np.max(np.array(std_scale_list) - 1),
    edgecolor="k",
    marker="o",
)
plt.colorbar()
plt.xlabel("n")
plt.ylabel("p")
plt.title("Standard deviation scale")
plt.tight_layout()
plt.savefig("error_heatmap.png")

quit()

# Statistical comparison
# Calculate descriptive statistics
binomial_mean = np.mean(binomial_sample)
binomial_std = np.std(binomial_sample)
normal_mean = np.mean(normal_sample)
normal_std = np.std(normal_sample)

print(f"Binomial Sample Mean: {binomial_mean}, Standard Deviation: {binomial_std}")
print(f"Normal Sample Mean: {normal_mean}, Standard Deviation: {normal_std}")

# Perform a Kolmogorov-Smirnov test
ks_statistic, p_value = stats.ks_2samp(binomial_sample, normal_sample)
print(f"KS Statistic: {ks_statistic}, P-value: {p_value}")

# Interpretation
if p_value > 0.05:
    print("The two samples are similar (fail to reject H0).")
else:
    print("The two samples are different (reject H0).")


# sort both samples and calculate the root mean squared difference
binomial_sample.sort()
normal_sample.sort()
rmsd = np.sqrt(np.mean((binomial_sample - normal_sample) ** 2))
print(f"Root Mean Squared Difference: {rmsd}")


# Visual comparison
plt.figure(figsize=(12, 10))

# Histogram of binomial sample
plt.hist(
    binomial_sample,
    bins=plot_max + 1,
    range=(-0.5, plot_max + 0.5),
    density=True,
    alpha=0.5,
    color="b",
    label="Binomial",
)
plt.hist(
    normal_sample,
    bins=plot_max + 1,
    range=(-0.5, plot_max + 0.5),
    density=True,
    alpha=0.5,
    color="r",
    label="Normal",
)
# set the y ticks every 0.1
plt.yticks(np.arange(0, 1.1, 0.1))
plt.grid()

plt.legend()
plt.ylim(0, 1)
plt.xlim(-0.5, plot_max + 0.5)
plt.title("Binomial Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
