import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import trapezoid
from scipy.constants import k, N_A
from optimize_argon_dimer import lennard_jones


#%%
#equations!
# Hard-Sphere Potential
def hard_sphere_potential(r, sigma=3.4):
    """
    Hard-sphere potential where the potential is infinite inside the diameter and zero outside.

    Parameters:
    r (float): Distance between two particles
    sigma (float): Diameter of the hard sphere (default is 3.4 Å)
    
    Returns:
    float: Hard-sphere potential
    """
    if r < sigma:
        return np.inf
    else:
        return 0

# Square-Well Potential
def square_well_potential(r, sigma=3.4, epsilon=0.01, lamb=1.5):
    """
    Square-well potential where the potential is infinite for r < sigma, negative within the well, and zero otherwise.

    Parameters:
    r (float): Distance between two particles
    sigma (float): Diameter of the particle (default is 3.4 Å)
    epsilon (float): Well depth (default is 0.01 eV)
    lamb (float): Well range (default is 1.5 times sigma)
    
    Returns:
    float: Square-well potential in Joules
    """
    if r < sigma:
        return np.inf
    elif sigma <= r < lamb * sigma:
        return -epsilon*1.60218e-19
    else:
        return 0

# Lennard-Jones Potential (reuse from Problem 1! but now in joules!)
def lennard_jones_potential(r, epsilon=0.01, sigma=3.4):
    """
    Computes the Lennard-Jones potential energy between two atoms given their interatomic distance, epsilon, and sigma.

    Parameters:
    r (float): Distance between two atoms (in Å).
    epsilon (float): Depth of the potential well (default is 0.01 eV).
    sigma (float): Distance at which the potential is zero (default is 3.4 Å).
    
    Returns:
    float: Lennard-Jones potential energy at distance r but in terms of Joules.
    """
    V = lennard_jones(r) * 1.60218e-19 #converts V(r) from eV to J
    return V

#%%
#def of B2v(T)
#some contants 

sigma=3.4
epsilon = 0.01 

def compute_b2v(potential_func, T, r_min = 0.001, r_max = 5*sigma , points=1000):
    """
    Computes the second virial coefficient B2V for a given potential using numerical integration (trapezoidal rule).

    Parameters:
    potential_func (function): The potential function (hard_sphere_potential, square_well_potential, lennard_jones_potential)
    T (float): Temperature in Kelvin
    sigma (float): Characteristic length 
    epsilon (float): Well depth for potentials 
    r_min(float): Lower bound for imtegration 
    r_max (float): Upper bound for integration (5 times sigma)
    points (int): Number of points for numerical integration 
    
    Returns:
    float: Second virial coefficient B2V
    """
    r_values = np.linspace(r_min, r_max, points)  
    integrand = []
    
#4.938475e+25
# B2V at 100K for Hard-Sphere: -0.000000e+00 m^3/mol but need to get #4.938475e+25
# B2V at 100K for Square-Well: -1.172839e+31 m^3/mol
# B2V at 100K for Lennard-Jones: -6.147197e+32 m^3/mol

    for r in r_values:
        u_r = potential_func(r)
        
        exp_term = np.exp(-u_r / (k * T))  # Compute exponential term
       
        integrand.append((exp_term - 1) * r**2)

    # Perform the integration using the trapezoidal rule
    integral_value = trapezoid(integrand, r_values)
    B2V = -2 * np.pi * N_A * integral_value
    return B2V

#%%
#computing B2V at 100K for each potential 
T = 100 #temp in K 

# Hard-Sphere Potential
b2v_hard_sphere = compute_b2v(hard_sphere_potential, T)

# Square-Well Potential
b2v_square_well = compute_b2v(square_well_potential, T)

# Lennard-Jones Potential
b2v_lennard_jones = compute_b2v(lennard_jones_potential, T)

print(f"B2V at 100K for Hard-Sphere: {b2v_hard_sphere:.6e} m^3/mol")
print(f"B2V at 100K for Square-Well: {b2v_square_well:.6e} m^3/mol")
print(f"B2V at 100K for Lennard-Jones: {b2v_lennard_jones:.6e} m^3/mol")

# %% 
#plot for temps from 100K to 800K 
def compute_b2v_vs_temperature(potential_func, temps):
    """
    Computes B2V over a range of temperatures for a given potential.

    Parameters:
    potential_func (function): The potential function
    temps (array): Array of temperatures
    
    Returns:
    list: List of B2V values corresponding to the temperatures
    """
    return [compute_b2v(potential_func, T) for T in temps]

# Temperature range from 100 K to 800 K
temps = np.linspace(100, 800, 100)

# Compute B2V for each potential
b2v_hard_sphere_vs_temp = compute_b2v_vs_temperature(hard_sphere_potential, temps)
b2v_square_well_vs_temp = compute_b2v_vs_temperature(square_well_potential, temps)
b2v_lennard_jones_vs_temp = compute_b2v_vs_temperature(lennard_jones_potential, temps)

# Plotting B2V vs Temperature
plt.plot(temps, b2v_hard_sphere_vs_temp, label="Hard-Sphere")
plt.plot(temps, b2v_square_well_vs_temp, label="Square-Well")
plt.plot(temps, b2v_lennard_jones_vs_temp, label="Lennard-Jones")

plt.axhline(0, color='black', linestyle='--')  # Line at B2V = 0
plt.xlabel(r'Temperature (K)')
plt.ylabel(r'$B_{2V} \, (\mathrm{m}^3/\mathrm{mol})$')
plt.title(r'Second Virial Coefficient $B_{2V}$ vs Temperature')
plt.legend()
plt.savefig('homework-2-2/b2v_vs_temperature.png')
plt.show()

#%%
#saving my results as a CSV file 
df = pd.DataFrame({
    'Temperature (K)': temps,
    'B2V Hard-Sphere (m^3/mol)': b2v_hard_sphere_vs_temp,
    'B2V Square-Well (m^3/mol)': b2v_square_well_vs_temp,
    'B2V Lennard-Jones (m^3/mol)': b2v_lennard_jones_vs_temp
})

df.to_csv('homework-2-2/b2v_results.csv', index=False)

#%%
#Discussion 
