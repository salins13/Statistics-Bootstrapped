import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

# Step 1: Generate a population with mean=50, SD=5, size=100,000
#np.random.seed(42)  # For reproducibility
population = np.random.normal(loc=50, scale=5, size=100000)

# Function to sample, compute SE, and CI
def analyze_sample(sample_size, num_samples=10):
    means = []
    cis = []
    samples = []  # Store samples for histogram
    ci_exclude_count = 0  # Counter for CIs that do not include population mean
    population_mean = np.mean(population)
    
    for _ in range(num_samples):
        sample = np.random.choice(population, size=sample_size, replace=False)
        samples.append(sample)
        sample_mean = np.mean(sample)
        sample_sd = np.std(sample, ddof=1)  # Sample standard deviation
        sample_se = sample_sd / np.sqrt(sample_size)  # Standard error
        
        # Compute 95% CI
        z_critical = 1.96  # Z-score for 95% confidence level
        ci_lower = sample_mean - z_critical * sample_se
        ci_upper = sample_mean + z_critical * sample_se
        
        means.append(sample_mean)
        cis.append((ci_lower, ci_upper))
        
        # Check if CI excludes population mean
        if ci_lower > population_mean or ci_upper < population_mean:
            ci_exclude_count += 1
    
    return means, cis, ci_exclude_count, samples

# Step 2: Run for different sample sizes with multiple samples
sample_sizes = [500, 50, 10]  # Large, medium, and small samples
num_samples = 100  # Number of repeated samplings
results = {size: analyze_sample(size, num_samples) for size in sample_sizes}

# Step 3: Smoothed Density Plots for Sample Distributions with Population Density Curve
fig, axes = plt.subplots(1, len(sample_sizes), figsize=(18, 6), sharey=True)
colors = sns.color_palette("bright", num_samples)  # Use distinct colors for each sample

for i, size in enumerate(sample_sizes):
    _, _, _, samples = results[size]
    
    for j, sample in enumerate(samples):
        sns.kdeplot(sample, color=colors[j], alpha=0.2, fill=True, ax=axes[i])
    
    # Plot population density curve
    sns.kdeplot(population, color='black', linewidth=2, label="Population Density", ax=axes[i])
    
    axes[i].axvline(np.mean(population), color='black', linestyle='dashed', label="Population Mean")
    axes[i].set_title(f"Smoothed Density of Samples (n={size})")
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Density")
    axes[i].legend()

plt.tight_layout()
plt.show()

# Step 4: Visualization of Sample Means and Confidence Intervals
fig, axes = plt.subplots(1, len(sample_sizes), figsize=(18, 6), sharey=True)
colors = ['blue', 'red', 'green']

for i, size in enumerate(sample_sizes):
    means, cis, ci_exclude_count, _ = results[size]
    y_positions = np.arange(num_samples)
    
    for j in range(num_samples):
        ci_lower, ci_upper = cis[j]
        axes[i].plot([ci_lower, ci_upper], [y_positions[j], y_positions[j]], color=colors[i], alpha=0.5)
        axes[i].scatter(means[j], y_positions[j], color=colors[i], label=f"Sample Mean (n={size})" if j == 0 else "")
    
    axes[i].axvline(np.mean(population), color='black', linestyle='dashed', label="Population Mean")
    axes[i].set_title(f"Sample Means with 95% Confidence Intervals (n={size})\nCIs not containing Population Mean: {ci_exclude_count}/{num_samples}")
    axes[i].set_xlabel("Mean Value")
    axes[i].set_ylabel("Sample Number")
    axes[i].legend()

plt.tight_layout()
plt.show()
