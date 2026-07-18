```python
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# ============================================
# LOAD DATA
# ============================================

try:
    df = pd.read_csv("data/finflow_users.csv")
except FileNotFoundError:
    print("Error: data/finflow_users.csv not found.")
    raise

session_minutes = df["session_minutes"].values
score_views = df["score_views"].values

# ============================================
# PART 1: MOMENTS & SHAPE ANALYSIS
# ============================================

mean_minutes = np.mean(session_minutes)

variance_minutes = np.var(session_minutes, ddof=1)

skewness_minutes = stats.skew(
    session_minutes,
    bias=False
)

kurtosis_minutes = stats.kurtosis(
    session_minutes,
    bias=False
)

# ============================================
# PART 2: DISTRIBUTION FITTING
# ============================================

# Poisson MLE
lambda_poisson = np.mean(score_views)

# Normal MLE
mu_normal, sigma_normal = stats.norm.fit(
    session_minutes
)

# KS Tests
ks_stat_poisson, p_value_poisson = stats.kstest(
    score_views,
    "poisson",
    args=(lambda_poisson,)
)

ks_stat_normal, p_value_normal = stats.kstest(
    session_minutes,
    "norm",
    args=(mu_normal, sigma_normal)
)

# ============================================
# PART 3: CLT SIMULATION
# ============================================

np.random.seed(42)

pop_mean = np.mean(session_minutes)
pop_std = np.std(session_minutes, ddof=1)

sample_sizes = [10, 30, 100]
n_reps = 10000

sampling_distributions = {}

for n in sample_sizes:

    samples = np.random.choice(
        session_minutes,
        size=(n_reps, n),
        replace=True
    )

    sample_means = samples.mean(axis=1)

    sampling_distributions[n] = sample_means

empirical_ses = {}

for n, means in sampling_distributions.items():
    empirical_ses[n] = np.std(
        means,
        ddof=1
    )

theoretical_ses = {
    n: pop_std / np.sqrt(n)
    for n in sample_sizes
}

min_n_normal = None

for n, means in sampling_distributions.items():

    skew_val = stats.skew(
        means,
        bias=False
    )

    if abs(skew_val) < 0.5:
        min_n_normal = n
        break

# ============================================
# VALIDATION
# ============================================

assert mean_minutes > 0
assert variance_minutes > 0
assert lambda_poisson > 0
assert sigma_normal > 0

for n in sample_sizes:

    rel_error = abs(
        empirical_ses[n]
        - theoretical_ses[n]
    ) / theoretical_ses[n]

    assert rel_error < 0.10

# ============================================
# RESULTS
# ============================================

print("SESSION DURATION MOMENTS")
print("=" * 50)

print(
    f"Mean: {mean_minutes:.2f} minutes"
)

print(
    f"Variance: {variance_minutes:.2f}"
)

print(
    f"Skewness: {skewness_minutes:.2f}"
)

print(
    f"Kurtosis: {kurtosis_minutes:.2f}"
)

print("\nSHAPE INTERPRETATION:")

if skewness_minutes > 0:
    skew_msg = (
        "Right-skewed distribution."
    )
else:
    skew_msg = (
        "Left-skewed distribution."
    )

print(
    f"Skewness ({skewness_minutes:.2f}): "
    f"{skew_msg}"
)

if kurtosis_minutes > 0:
    kurt_msg = (
        "Heavy tails and more extreme outliers."
    )
else:
    kurt_msg = (
        "Lighter tails than Normal."
    )

print(
    f"Kurtosis ({kurtosis_minutes:.2f}): "
    f"{kurt_msg}"
)

print("\nBUSINESS IMPLICATION:")
print(
    "A small number of users spend "
    "significantly longer in sessions, "
    "suggesting opportunities to study "
    "power-user behaviour."
)

# ============================================
# DISTRIBUTION RESULTS
# ============================================

print("\n" + "=" * 60)
print("DISTRIBUTION FITTING RESULTS")
print("=" * 60)

print(
    f"{'Distribution':<15}"
    f"{'Parameters':<25}"
    f"{'KS Stat':<10}"
    f"{'p-value'}"
)

print("-" * 60)

print(
    f"Poisson{'':<8}"
    f"λ={lambda_poisson:.2f}{'':<18}"
    f"{ks_stat_poisson:.3f}"
    f"{'':<6}"
    f"{p_value_poisson:.3f}"
)

print(
    f"Normal{'':<9}"
    f"μ={mu_normal:.2f}, "
    f"σ={sigma_normal:.2f}"
    f"{'':<8}"
    f"{ks_stat_normal:.3f}"
    f"{'':<6}"
    f"{p_value_normal:.3f}"
)

print("=" * 60)

print("\nGOODNESS-OF-FIT INTERPRETATION:")

if p_value_poisson < 0.05:
    print(
        "Poisson fit: Poor fit "
        "(reject H0)."
    )
else:
    print(
        "Poisson fit: Acceptable fit."
    )

if p_value_normal < 0.05:
    print(
        "Normal fit: Poor fit "
        "(reject H0)."
    )
else:
    print(
        "Normal fit: Acceptable fit."
    )

print("\nRECOMMENDATION:")
print(
    "Use Poisson models for count data "
    "such as score views and Normal "
    "approximations for sampling "
    "distributions of means."
)

# ============================================
# CLT RESULTS
# ============================================

print("\n" + "=" * 60)
print("CLT CONVERGENCE RESULTS")
print("=" * 60)

print(
    f"{'Sample Size':<20}"
    f"{'Empirical SE':<18}"
    f"{'Theoretical SE':<18}"
    f"{'Ratio'}"
)

print("-" * 60)

for n in sample_sizes:

    ratio = (
        empirical_ses[n]
        / theoretical_ses[n]
    )

    print(
        f"{n:<20}"
        f"{empirical_ses[n]:<18.2f}"
        f"{theoretical_ses[n]:<18.2f}"
        f"{ratio:.3f}"
    )

print("=" * 60)

print(
    f"\nMinimum n for approximate "
    f"Normality: {min_n_normal}"
)

print("\nBUSINESS IMPLICATION:")

print(
    f"For A/B testing, a minimum sample "
    f"size of approximately "
    f"{min_n_normal} users per group "
    f"is recommended because the "
    f"sampling distribution becomes "
    f"approximately Normal."
)

# Optional QQ Plot
plt.figure(figsize=(6, 6))
stats.probplot(
    session_minutes,
    dist="norm",
    plot=plt
)
plt.title(
    "Q-Q Plot: Session Minutes"
)
plt.tight_layout()
plt.savefig("qq_plot.png")
```

