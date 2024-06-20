import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

n_arr = np.arange(10, 1000, 10)
p_arr = np.array([0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999])

### TODO I have the problem that for very small p the normal distribution is not a good
### approximation of the binomial distribution.
### I think one can shift the mean and scale the standard deviation depending on the p
### and n values. I will try to optimize the shift and scale for each n and p value.
shift_mean_bounds = [-1, 1]
scale_std_bounds = [0.5, 2]

for n in n_arr:
    for p in p_arr:
        # set n and p
        # then optimize shift of mean
        # Parameters
        # number of samples
        N = 10000

        # Generate data samples
        binomial_sample = np.random.binomial(n, p, N)
        mean = n * p
        std_dev = np.sqrt(n * p * (1 - p))
        normal_sample = np.random.normal(mean, std_dev, N)

        ### round and clip the normal sample
        normal_sample = np.round(normal_sample)
        normal_sample[normal_sample < 0] = 0
        normal_sample[normal_sample > n] = n

        print(np.histogram(binomial_sample, bins=n + 1, range=(-0.5, n + 0.5))[0])
        print(np.histogram(normal_sample, bins=n + 1, range=(-0.5, n + 0.5))[0])
        diff = (
            np.histogram(binomial_sample, bins=n + 1, range=(-0.5, n + 0.5))[0]
            - np.histogram(normal_sample, bins=n + 1, range=(-0.5, n + 0.5))[0]
        )
        error = np.sum(np.abs(diff))

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
