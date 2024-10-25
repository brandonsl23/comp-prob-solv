# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon
import nbformat as nbf
import io
import base64

## Setting parameters:
a0 = 1  
L = 20.0 
V = (2*L)**3 # volume of integration region

np.random.seed(42)
# Create a new Jupyter notebook object
notebook = nbf.v4.new_notebook()

# Add a text cell to the notebook
intro_text = """# Figures and Dissussion from Random and Importance Sampling

This notebook contains the figures generated from the sampling methods for overlap integrals.
"""
notebook.cells.append(nbf.v4.new_markdown_cell(intro_text))

# Function to save the current figure as base64 string, this allows for it be put into markdown in jupyter notebooks!
def save_fig_to_base64():
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode("utf-8")
    return img_str

# %%
# 2p orbital function
def psi_2p_z(x, y, z):
    """
    Computes the value of the hydrogen 2p orbital oriented along the z-axis at a given point in Cartesian coordinates.

    Parameters:
    x (float): x-coordinate of the point.
    y (float): y-coordinate of the point.
    z (float): z-coordinate of the point.

    Returns:
    float: Value of the 2p orbital at the given point.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    c = 1 / (4 * np.sqrt(2 * np.pi) * a0**(3/2))
    return c * (r / a0) * (z / r) * np.exp(-r / (2 * a0))

# %%
# integrand for the overlap integral
def integrand(x, y, z, R):
    """
    Computes the integrand for the overlap integral of two hydrogen 2p orbitals separated by distance R along the z-axis.

    Parameters:
    x (float): x-coordinate of the point.
    y (float): y-coordinate of the point.
    z (float): z-coordinate of the point.
    R (float): Separation distance between the two 2p orbitals (in atomic units).

    Returns:
    float: Value of the integrand at the given point.
    """
    psi_plus = psi_2p_z(x, y, z + R/2)
    psi_minus = psi_2p_z(x, y, z - R/2)
    return psi_plus * psi_minus

# %% Monte Carlo integration with random sampling
def monte_carlo_overlap(N, R):
    """
    Performs Monte Carlo integration using random sampling to estimate the overlap integral S(R) between two hydrogen 2p orbitals.

    Parameters:
    N (int): Number of random points to sample.
    R (float): Separation distance between the two 2p orbitals (in atomic units).

    Returns:
    float: Estimated overlap integral S(R) based on random sampling.
    """
    x = np.random.uniform(-L, L, N)
    y = np.random.uniform(-L, L, N)
    z = np.random.uniform(-L, L, N)
    integrand_values = integrand(x, y, z, R)
    average_integrand = np.mean(integrand_values)
    S_R = V * average_integrand
    return S_R

# %% Monte Carlo integration with importance sampling
def monte_carlo_importance_sampling(N, R):
    """
    Performs Monte Carlo integration using importance sampling to estimate the overlap integral S(R) between two hydrogen 2p orbitals.

    Parameters:
    N (int): Number of random points to sample.
    R (float): Separation distance between the two 2p orbitals (in atomic units).

    Returns:
    float: Estimated overlap integral S(R) using importance sampling.
    """
    x = expon.rvs(size=N, scale=1.0)
    y = expon.rvs(size=N, scale=1.0)
    z = expon.rvs(size=N, scale=1.0)
    numerator = integrand(x, y, z, R)
    denominator = expon.pdf(x) * expon.pdf(y) * expon.pdf(z)
    integrand_values = numerator / denominator
    average_integrand = np.mean(integrand_values)
    S_R = average_integrand * 8  # Estimate of the overlap integral in 8 octants
    return S_R

# %% Range of N values for comparison
N_values = [10**2, 10**3, 10**4, 10**5, 10**6, 10**7, 10**8]
S_R_estimates = []

# Perform Monte Carlo integration with random sampling for each N
for N in N_values:
    S_R = monte_carlo_overlap(N, R=2.0)
    S_R_estimates.append(S_R)
    print(f" N = {N}, S(R) ≈ {S_R}")

#%% 
## random sampling graph 
plt.figure(figsize=(8, 6))
plt.plot(N_values, S_R_estimates, label="Random Sampling")
plt.xscale('log')
plt.xlabel('Number of Points Sampled (N)')
plt.ylabel('Integral Value S(R)')
plt.title('Random Sampling S(R) vs N')
plt.legend()
plt.grid(True)

# Save figure to notebook
img_str = save_fig_to_base64()
img_md = f"![Random Sampling Plot](data:image/png;base64,{img_str})"
notebook.cells.append(nbf.v4.new_markdown_cell(img_md))

#Adding discussion for random sampling  
rand_text = """## Dissussion for Random Sampling
The graph shows that random sampling exhibits significant fluctuations when N is small. 
However, as N increases,(e.g., N = 10^6 and beyond), the estimated value of S(R)
appears to stabilize around a consistent value (roughly around S(R)≈0.73)."""

notebook.cells.append(nbf.v4.new_markdown_cell(rand_text))

plt.show()

# %% Importance sampling
S_R_importance_estimates = []
for N in N_values:
    S_R = monte_carlo_importance_sampling(N, R=2.0)
    S_R_importance_estimates.append(S_R)
    print(f"N = {N}, S(R) ≈ {S_R}")

#%%
## graphing importance sampling 
plt.figure(figsize=(8, 6))
plt.plot(N_values, S_R_importance_estimates, color="orange", label="Importance Sampling")
plt.xscale('log')
plt.xlabel('Number of Points Sampled (N)')
plt.ylabel('Integral Value S(R))')
plt.title('Importance Sampling S(R) vs N')
plt.legend()
plt.grid(True)

# Save figure to notebook
img_str = save_fig_to_base64()
img_md = f"![Importance Sampling Plot](data:image/png;base64,{img_str})"
notebook.cells.append(nbf.v4.new_markdown_cell(img_md))

#Adding discussion for random sampling  
imp_text = """## Dissussion for Importance Sampling vs Random Sampling
The graph demonstrates that the integral value S(R) converges rapidly, with the estimate stabilizing 
around N=10^4.The final value of S(R) approaches approximately 0.73, aligning closely with the value
observed in random sampling for higher N. This consistency between the two methods at large sample sizes 
indicates that both approaches are converging to the correct result, though importance sampling reaches this 
conclusion significantly faster. Importantly, importance sampling proves to be far more efficient than random 
sampling, especially when working with smaller sample sizes. It provides more accurate and reliable estimates 
with fewer points, primarily due to the reduced variance inherent in the sampling process.
"""
notebook.cells.append(nbf.v4.new_markdown_cell(imp_text))


# %% Plot comparison of random sampling and importance sampling
plt.figure(figsize=(8, 6))
plt.plot(N_values, S_R_estimates, label="Random Sampling")
plt.plot(N_values, S_R_importance_estimates, label="Importance Sampling")
plt.xscale('log')
plt.xlabel('Number of Points Sampled (N)')
plt.ylabel('Estimated Overlap Integral S(R)')
plt.title('Comparison of Random and Importance Sampling')
plt.legend()
plt.grid(True)

# Save figure to notebook
img_str = save_fig_to_base64()
img_md = f"![Comparison Plot](data:image/png;base64,{img_str})"
notebook.cells.append(nbf.v4.new_markdown_cell(img_md))


# %% Overlap integral as a function of separation distance
R_values = np.arange(0.5, 20.5, 0.5)  # From 0.5 to 20 in increments of 0.5
N = 10**6  # Fixed number of sampling points for importance sampling

S_R_range_importance = []

for R in R_values:
    S_R = monte_carlo_importance_sampling(N, R)
    S_R_range_importance.append(S_R)

# Plot S(R) vs R
plt.figure(figsize=(8, 6))
plt.plot(R_values, S_R_range_importance)
plt.xlabel('Separation Distance R (atomic units)')
plt.ylabel('Overlap Integral S(R)')
plt.title('Overlap Integral S(R) vs Separation Distance R')
plt.grid(True)

# Save figure to notebook
img_str = save_fig_to_base64()
img_md = f"![S(R) vs R Plot](data:image/png;base64,{img_str})"
notebook.cells.append(nbf.v4.new_markdown_cell(img_md))

sep_text = """## Dissussion for Overlap Integral as a Function of Separation Distance

The change in the sign of S(R) occurs because of the phase relationship between the 2p orbitals as the separation distance 
R increases. The orbitals are in phase at small R, resulting in a positive overlap. As R increases, the orbitals move out of phase, 
leading to a negative overlap. At large distances, the overlap becomes minimal, and S(R) approaches zero as the interaction between the orbitals fades.
"""
notebook.cells.append(nbf.v4.new_markdown_cell(sep_text))

plt.show()

# %% Save the notebook to a file
with open("sampling_plots_notebook.ipynb", "w") as f:
    nbf.write(notebook, f)

print("Notebook generated as 'sampling_plots_notebook.ipynb'")

# %%

