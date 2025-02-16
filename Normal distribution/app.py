import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


# Generate a population with mean=50, SD=5, size=100,000
population = np.random.normal(loc=50, scale=5, size=100000)

# Function to sample, compute SE, and CI
def analyze_sample(sample_size, num_samples=10, z_critical=1.96):
    means = []
    cis = []
    samples = []  # Store samples for visualization
    ci_exclude_count = 0  # Counter for CIs that do not include population mean
    population_mean = np.mean(population)
    
    for _ in range(num_samples):
        sample = np.random.choice(population, size=sample_size, replace=False)
        samples.append(sample)
        sample_mean = np.mean(sample)
        sample_sd = np.std(sample, ddof=1)  # Sample standard deviation
        sample_se = sample_sd / np.sqrt(sample_size)  # Standard error
        
        # Compute CI
        ci_lower = sample_mean - z_critical * sample_se
        ci_upper = sample_mean + z_critical * sample_se
        
        means.append(sample_mean)
        cis.append((ci_lower, ci_upper))
        
        # Check if CI excludes population mean
        if ci_lower > population_mean or ci_upper < population_mean:
            ci_exclude_count += 1
    
    return means, cis, ci_exclude_count, samples

# Streamlit UI
st.title("Confidence Interval Visualization App")

# User inputs
sample_sizes = st.text_input("Enter up to 4 sample sizes separated by commas (e.g., 500,50,10)", "500,50,10")
sample_sizes = list(map(int, sample_sizes.split(",")))[:4]  # Limit to 4 sample sizes
num_samples = st.number_input("Enter the number of samples to be taken", min_value=1, max_value=100, value=10)
confidence_level = st.slider("Select confidence level", 0.80, 0.99, 0.95, step=0.01)
z_critical = stats.norm.ppf(1 - (1 - confidence_level) / 2)

# Compute results
results = {size: analyze_sample(size, num_samples, z_critical) for size in sample_sizes}

# Visualization - Smoothed Density Plots
st.subheader("Sample Distributions with Population Density Curve")
fig, axes = plt.subplots(1, len(sample_sizes), figsize=(18, 6), sharey=True)
if len(sample_sizes) == 1:
    axes = [axes]  # Ensure axes is always a list
colors = sns.color_palette("bright", num_samples)

for i, size in enumerate(sample_sizes):
    _, _, _, samples = results[size]
    for j, sample in enumerate(samples):
        sns.kdeplot(sample, color=colors[j % len(colors)], alpha=0.2, fill=True, ax=axes[i])
    sns.kdeplot(population, color='black', linewidth=2, label="Population Density", ax=axes[i])
    axes[i].axvline(np.mean(population), color='black', linestyle='dashed', label="Population Mean")
    axes[i].set_title(f"Smoothed Density of Samples (n={size})")
    axes[i].set_xlabel("Value")
    axes[i].set_ylabel("Density")
    axes[i].legend()

st.pyplot(fig)

# Visualization - Sample Means and Confidence Intervals
st.subheader("Sample Means with Confidence Intervals")
fig, axes = plt.subplots(1, len(sample_sizes), figsize=(18, 6), sharey=True)
if len(sample_sizes) == 1:
    axes = [axes]
colors = sns.color_palette("bright", len(sample_sizes))

for i, size in enumerate(sample_sizes):
    means, cis, ci_exclude_count, _ = results[size]
    y_positions = np.arange(num_samples)
    for j in range(num_samples):
        ci_lower, ci_upper = cis[j]
        axes[i].plot([ci_lower, ci_upper], [y_positions[j], y_positions[j]], color=colors[i], alpha=0.5)
        axes[i].scatter(means[j], y_positions[j], color=colors[i], label=f"Sample Mean (n={size})" if j == 0 else "")
    axes[i].axvline(np.mean(population), color='black', linestyle='dashed', label="Population Mean")
    axes[i].set_title(f"Sample Means with {confidence_level*100}% CI (n={size})\nCIs not containing Population Mean: {ci_exclude_count}/{num_samples}")
    axes[i].set_xlabel("Mean Value")
    axes[i].set_ylabel("Sample Number")
    axes[i].legend()

st.pyplot(fig)
