#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t


# %%
# This is from Lecture 7!
# Function to compute OLS slope
def ols_slope(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    return numerator / denominator

# Function to compute OLS intercept
def ols_intercept(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    slope = ols_slope(x, y)
    return y_mean - slope * x_mean

# Function to compute both OLS slope and intercept
def ols(x, y):
    slope = ols_slope(x, y)
    intercept = ols_intercept(x, y)
    return slope, intercept

# %%
df = pd.read_csv('trouton.csv')
df.head()

# %%
# Converting H_v from kcal/mol to J/mol
df['H_v (J/mol)'] = df['H_v (kcal/mol)'] * 4184

# Extracting data from CSV file 
t_b = df['T_B (K)']
h_v = df['H_v (J/mol)']

# %% 
# OLS regression
slope, intercept = ols(t_b, h_v)
regression_line = t_b * slope + intercept


# %%
# Calculate the 95% confidence intervals for slope (a) from Lecture 8! 

# Calculate the residuals
residuals = h_v - regression_line

# Calculate the sum of squared residuals
def sse(residuals):
    return np.sum(residuals ** 2)

# Calculate the variance of the residuals
def variance(residuals):
    return sse(residuals) / (len(residuals) - 2)

# Calculate the standard error of the slope
def se_slope(x, residuals):
    numerator = variance(residuals)
    x_mean = np.mean(x)
    denominator = np.sum((x - x_mean) ** 2)
    return np.sqrt(numerator / denominator)

# Calculate the confidence interval for the slope
def confidence_interval_slope(x, residuals, confidence_level):
    se = se_slope(x, residuals)
    df = len(x) - 2  # degrees of freedom
    alpha = 1 - confidence_level
    critical_t_value = t.ppf(1 - alpha/2, df)
    return critical_t_value * se

# Calculate the 95% confidence interval for the slope
conf_slope = confidence_interval_slope(t_b, residuals, 0.95)
print(f"Slope: {slope:.3f} +/- {conf_slope:.3f}")

# %%
# Calculate the 95% confidence intervals for the intercept (b) from Lecture 8!

# Calculate the standard error of the intercept
def se_intercept(x, residuals):
    numerator = variance(residuals)
    x_mean = np.mean(x)
    denominator = len(x) * np.sum((x - x_mean) ** 2)
    return np.sqrt(numerator / denominator)

# Calculate the confidence interval for the intercept
def confidence_interval_intercept(x, residuals, confidence_level):
    se = se_intercept(x, residuals)
    df = len(x) - 2  # degrees of freedom
    alpha = 1 - confidence_level
    critical_t_value = t.ppf(1 - alpha/2, df)
    return critical_t_value * se

# Calculate the 95% confidence interval for the intercept
conf_intercept = confidence_interval_intercept(t_b, residuals, 0.95)
print(f"Intercept: {intercept:.3f} +/- {conf_intercept:.3f}")


# %%
# Create plot with linear regression line
plt.plot(t_b, regression_line, color='red',label=f'Fit: $H_v = ({slope:.3f} \pm {conf_slope:.3f}) T_B + ({intercept/1000:.3f} \pm {conf_intercept/1000:.3f})$ J/mol')

# Plotting residuals 
for i in range(len(t_b)):
    plt.plot([t_b[i], t_b[i]], [h_v[i], regression_line[i]], color='gray')
# Plot the data with colors by class
classes = df['Class'].unique()
colors = plt.cm.viridis(np.linspace(0, 1, len(classes)))

for i, class_name in enumerate(classes):
    class_data = df[df['Class'] == class_name]
    plt.scatter(class_data['T_B (K)'], class_data['H_v (J/mol)'], color=colors[i], label=class_name)

# Add labels, title, and grid
plt.xlabel('$T_B (K)$')
plt.ylabel('$H_v (J/mol)$')
plt.title('Trouton’s Rule')
plt.legend(loc='best')
plt.grid(True)



# Ensure directory exists and save plot
plt.savefig('homework-3-1/troutons_rule_plot.png')
plt.show()


# %%
#Discussion on Troutons rule 
#The computed entropy of vaporization (Δ S_v​) from the dataset is 103.85 J/mol-K, 
#which is higher than the 88 J/mol-K approximation given by Trouton’s Rule. This deviation can 
#likely be attributed to the presence of substances in the dataset that exhibit stronger imfs, such as hydrogen bonding (using methanol as an example on the list), 
#or other complex interactions, which increase the entropy of vaporization. Therefore, while Trouton’s Rule provides a useful approximation for non-associating 
#liquids, it does not fully apply to  all the substances in the dataset.
#
#
# %%
