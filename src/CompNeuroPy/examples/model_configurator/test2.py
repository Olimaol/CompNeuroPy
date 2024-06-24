import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from CompNeuroPy import DeapCma, save_variables, load_variables


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


def evaluate_function(population):
    loss_list = []
    ### the population is a list of individuals which are lists of parameters
    for individual in population:
        loss_of_individual = objective_function(
            mean_shift=individual[0], std_scale=individual[1]
        )
        loss_list.append((loss_of_individual,))
    return loss_list


n_arr = np.linspace(10, 1000, 20, dtype=int)
p_arr = np.array([0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999])

### TODO I have the problem that for very small p the normal distribution is not a good
### approximation of the binomial distribution.
### I think one can shift the mean and scale the standard deviation depending on the p
### and n values. I will try to optimize the shift and scale for each n and p value.

### bounds for optimized parameters
shift_mean_bounds = [-1, 1]
scale_std_bounds = [0.5, 2]
lb = np.array([shift_mean_bounds[0], scale_std_bounds[0]])
ub = np.array([shift_mean_bounds[1], scale_std_bounds[1]])
# number of samples
M = 10000
OPTIMIZE = False
if OPTIMIZE:
    p_list = []
    n_list = []
    mean_shift_list = []
    std_scale_list = []
    error_list = []
    error_improved_list = []
    for p in p_arr:
        for n in n_arr:
            ### set the global variables probability of success and number of trials
            P = p
            N = n

            ### get the error without optimization
            error = objective_function(mean_shift=0, std_scale=1)
            error_list.append(error)

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
            error = deap_cma_result["best_fitness"]

            ### store the results
            p_list.append(p)
            n_list.append(n)
            error_improved_list.append(error)
            mean_shift_list.append(mean_shift)
            std_scale_list.append(std_scale)

    ### save variables
    save_variables(
        variable_list=[
            p_list,
            n_list,
            mean_shift_list,
            std_scale_list,
            error_list,
            error_improved_list,
        ],
        name_list=[
            "p_list",
            "n_list",
            "mean_shift_list",
            "std_scale_list",
            "error_list",
            "error_improved_list",
        ],
        path="data_optimize_binomial_normal",
    )
else:
    loaded_variables = load_variables(
        name_list=[
            "p_list",
            "n_list",
            "mean_shift_list",
            "std_scale_list",
            "error_list",
            "error_improved_list",
        ],
        path="data_optimize_binomial_normal",
    )
    p_list = loaded_variables["p_list"]
    n_list = loaded_variables["n_list"]
    mean_shift_list = loaded_variables["mean_shift_list"]
    std_scale_list = loaded_variables["std_scale_list"]
    error_list = loaded_variables["error_list"]
    error_improved_list = loaded_variables["error_improved_list"]


# Plot the error as a function of p and n as a heatmap
plt.figure(figsize=(12, 10))
### scatter plot with max from error_list and error_improved_list
plt.subplot(2, 2, 1)
plt.scatter(
    n_list,
    p_list,
    c=error_list,
    cmap="viridis",
    vmin=0,
    vmax=np.max([error_list, error_improved_list]),
)
plt.colorbar()
plt.xlabel("n")
plt.ylabel("p")
plt.title("Error original")
plt.subplot(2, 2, 2)
plt.scatter(
    n_list,
    p_list,
    c=error_improved_list,
    cmap="viridis",
    vmin=0,
    vmax=np.max([error_list, error_improved_list]),
)
plt.colorbar()
plt.xlabel("n")
plt.ylabel("p")
plt.title("Error optimized")
plt.subplot(2, 2, 3)
plt.scatter(n_list, p_list, c=mean_shift_list, cmap="viridis")
plt.colorbar()
plt.xlabel("n")
plt.ylabel("p")
plt.title("Mean shift")
plt.subplot(2, 2, 4)
plt.scatter(n_list, p_list, c=std_scale_list, cmap="viridis")
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
