import statsmodels.stats.power as smp

effect_size = 0.5  # Desired effect size
alpha = 0.05  # Significance level
num_groups = 6  # Number of groups

power_analysis = smp.FTestAnovaPower()
sample_size = power_analysis.solve_power(effect_size=effect_size, alpha=alpha, k_groups=num_groups, power=0.8)

print("Required Sample Size:", round(sample_size))