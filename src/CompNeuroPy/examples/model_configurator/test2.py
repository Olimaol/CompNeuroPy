import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Parameters
n = 10  # number of trials
p = 0.01  # probability of success
N = 10000  # number of samples

# Generate data samples
binomial_sample = np.random.binomial(n, p, N)
mean = n * p
std_dev = np.sqrt(n * p * (1 - p))
normal_sample = np.random.normal(mean, std_dev, N)

# ### scale normal sample above mean and below mean
# normal_sample_original = normal_sample.copy()
# normal_sample[normal_sample_original >= mean] = (
#     normal_sample_original[normal_sample_original >= mean] * 1.1
# )
# normal_sample[normal_sample_original < mean] = (
#     normal_sample_original[normal_sample_original < mean] * 0.9
# )

### round and clip the normal sample
normal_sample = np.round(normal_sample)
normal_sample[normal_sample < 0] = 0
normal_sample[normal_sample > n] = n


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
plt.figure(figsize=(12, 6))

# Histogram of binomial sample
plt.subplot(1, 2, 1)
plt.hist(
    binomial_sample,
    bins=n + 1,
    range=(-0.5, n + 0.5),
    density=True,
    alpha=0.5,
    color="b",
    label="Binomial",
)
plt.hist(
    binomial_sample,
    bins=n * 50,
    range=(-0.5, n + 0.5),
    density=True,
    alpha=0.5,
    color="b",
    label="Binomial",
)
plt.xlim(-0.5, n + 0.5)
plt.title("Binomial Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")

# Histogram of normal sample
plt.subplot(1, 2, 2)
plt.hist(
    normal_sample,
    bins=n + 1,
    range=(-0.5, n + 0.5),
    density=True,
    alpha=0.5,
    color="r",
    label="Normal",
)
plt.hist(
    normal_sample,
    bins=n * 50,
    range=(-0.5, n + 0.5),
    density=True,
    alpha=0.5,
    color="r",
    label="Normal",
)
plt.xlim(-0.5, n + 0.5)
plt.title("Normal Distribution")
plt.xlabel("Value")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
