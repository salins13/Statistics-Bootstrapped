import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import streamlit as st
from utils.analysis import analyze_sample

# Step 1: Generate a population with mean=50, SD=5, size=100,000
population = np.random.normal(loc=50, scale=5, size=100000)

# Streamlit application title
st.title("Sample Analysis with Confidence Intervals")

# Step 2: User input for sample sizes and number of samples
sample_sizes = st.text_input("Enter up to 4 sample sizes separated by commas (e.g., 500,50,10):", "500,50,10")
sample_sizes = list(map(int, sample_sizes.split(',')))[:4]

num_samples = st.number_input("Enter the number of samples to be taken:", min_value=1, value=10)

# Step 3: User input for confidence level
confidence_level = st.number_input("Enter the confidence level (e.g., 0.95):", min_value=0.0, max_value=1.0, value=0.95)
z_critical = stats.norm.ppf(1 - (1 - confidence_level) / 2)

# Step 4: Run analysis for different sample sizes
results = {size: analyze_sample(size, num_samples, z_critical) for size in sample_sizes}

# Step 5: Smoothed Density Plots for Sample Distributions with Population Density Curve
st.subheader("Smoothed Density of Samples")
for size in sample_sizes:
    _, _, _, samples = results[size]
    fig, ax = plt.subplots()
    
    colors = sns.color_palette("bright", num_samples)
    for j, sample in enumerate(samples):
        sns.kdeplot(sample, color=colors[j], alpha=0.2, fill=True, ax=ax)
    
    sns.kdeplot(population, color='black', linewidth=2, label="Population Density", ax=ax)
    ax.axvline(np.mean(population), color='black', linestyle='dashed', label="Population Mean")
    ax.set_title(f"Smoothed Density of Samples (n={size})")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.legend()
    st.pyplot(fig)

# Step 6: Visualization of Sample Means and Confidence Intervals
st.subheader("Sample Means with Confidence Intervals")
for size in sample_sizes:
    means, cis, ci_exclude_count, _ = results[size]
    fig, ax = plt.subplots()
    y_positions = np.arange(num_samples)
    
    colors = ['blue', 'red', 'green']
    for j in range(num_samples):
        ci_lower, ci_upper = cis[j]
        ax.plot([ci_lower, ci_upper], [y_positions[j], y_positions[j]], color=colors[0], alpha=0.5)
        ax.scatter(means[j], y_positions[j], color=colors[0], label=f"Sample Mean (n={size})" if j == 0 else "")
    
    ax.axvline(np.mean(population), color='black', linestyle='dashed', label="Population Mean")
    ax.set_title(f"Sample Means with {confidence_level*100}% Confidence Intervals (n={size})\nCIs not containing Population Mean: {ci_exclude_count}/{num_samples}")
    ax.set_xlabel("Mean Value")
    ax.set_ylabel("Sample Number")
    ax.legend()
    st.pyplot(fig)