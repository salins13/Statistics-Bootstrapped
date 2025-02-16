import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Function to generate population
def generate_population(mean=50, sd=5, size=100000, seed=42):
    np.random.seed(seed)
    return np.random.normal(loc=mean, scale=sd, size=size)

# Function to sample, compute SE, and CI
def analyze_sample(population, sample_size, num_samples, z_critical):
    means = []
    cis = []
    ci_exclude_count = 0  # Counter for CIs that do not include population mean
    population_mean = np.mean(population)
    samples = []
    
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
        
        if ci_lower > population_mean or ci_upper < population_mean:
            ci_exclude_count += 1
    
    return means, cis, ci_exclude_count, samples

# Streamlit UI
def main():
    st.title("Sampling Distribution and Confidence Intervals")
    
    # Sidebar options
    sample_sizes = st.multiselect("Select Sample Sizes:", [10, 100, 1000], default=[100])
    num_samples = st.slider("Number of Samples:", min_value=10, max_value=200, value=100)
    confidence_level = st.selectbox("Select Confidence Level:", [90, 95, 99], index=1)
    
    # Compute Z-score for selected confidence level
    z_critical = {90: 1.645, 95: 1.96, 99: 2.576}[confidence_level]
    
    # Generate population
    population = generate_population()
    population_mean = np.mean(population)
    
    results = {size: analyze_sample(population, size, num_samples, z_critical) for size in sample_sizes}
    
    # Step 1: Smoothed Density Plots
    st.subheader("Step 1: Sample Distributions vs Population Density")
    fig, axes = plt.subplots(1, len(sample_sizes), figsize=(18, 6), sharey=True)
    colors = sns.color_palette("husl", num_samples)  
    
    if len(sample_sizes) == 1:
        axes = [axes]
    
    for i, size in enumerate(sample_sizes):
        _, _, _, samples = results[size]
        
        for j, sample in enumerate(samples):
            sns.kdeplot(sample, color=colors[j], alpha=0.2, fill=True, ax=axes[i])
        
        sns.kdeplot(population, color='black', linewidth=2, label="Population Density", ax=axes[i])
        axes[i].axvline(population_mean, color='black', linestyle='dashed', label="Population Mean")
        axes[i].set_title(f"Smoothed Density of Samples (n={size})")
        axes[i].set_xlabel("Value")
        axes[i].legend()
    
    st.pyplot(fig, clear_figure=True)
    
    # Step 2: Sample Means with Confidence Intervals
    st.subheader("Step 2: Sample Means and Confidence Intervals")
    fig, axes = plt.subplots(1, len(sample_sizes), figsize=(18, 6), sharey=True)
    colors = ['blue', 'red', 'green']
    
    if len(sample_sizes) == 1:
        axes = [axes]
    
    for i, size in enumerate(sample_sizes):
        means, cis, ci_exclude_count, _ = results[size]
        y_positions = np.arange(num_samples)
        
        for j in range(num_samples):
            ci_lower, ci_upper = cis[j]
            axes[i].plot([ci_lower, ci_upper], [y_positions[j], y_positions[j]], color=colors[i % len(colors)], alpha=0.5)
            axes[i].scatter(means[j], y_positions[j], color=colors[i % len(colors)], label=f"Sample Mean (n={size})" if j == 0 else "")
        
        axes[i].axvline(population_mean, color='black', linestyle='dashed', label="Population Mean")
        axes[i].set_title(f"Sample Means with {confidence_level}% Confidence Intervals (n={size})\nCIs not containing Population Mean: {ci_exclude_count}/{num_samples}")
        axes[i].set_xlabel("Mean Value")
        axes[i].legend()
    
    st.pyplot(fig)

if __name__ == "__main__":
    main()
