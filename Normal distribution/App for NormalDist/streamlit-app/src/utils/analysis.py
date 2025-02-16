import numpy as np
import scipy.stats as stats

def analyze_sample(population, sample_size, num_samples=10, z_critical=1.96):
    means = []
    cis = []
    ci_exclude_count = 0
    population_mean = np.mean(population)
    
    for _ in range(num_samples):
        sample = np.random.choice(population, size=sample_size, replace=False)
        sample_mean = np.mean(sample)
        sample_sd = np.std(sample, ddof=1)
        sample_se = sample_sd / np.sqrt(sample_size)
        
        ci_lower = sample_mean - z_critical * sample_se
        ci_upper = sample_mean + z_critical * sample_se
        
        means.append(sample_mean)
        cis.append((ci_lower, ci_upper))
        
        if ci_lower > population_mean or ci_upper < population_mean:
            ci_exclude_count += 1
    
    return means, cis, ci_exclude_count