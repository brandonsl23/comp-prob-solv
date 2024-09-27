# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

# %%
def lennard_jones(r, epsilon=0.01, sigma=3.4):
    """
    Computes the Lennard-Jones potential energy between two atoms given their interatomic distance, epsilon, and sigma.

    Parameters:
    r (float): Distance between two atoms (in Å).
    epsilon (float): Depth of the potential well (default is 0.01 eV).
    sigma (float): Distance at which the potential is zero (default is 3.4 Å).
    
    Returns:
    float: Lennard-Jones potential energy at distance r.
    """
    V = 4*epsilon*((sigma/r)**12 - (sigma/r)**6)
    return V



# %%
def optimize_ar_dimer (initial_guess = 4,epsilon=0.01, sigma=3.4):
    """
    Optimizes the distance between two argon atoms to minimize the Lennard-Jones potential.

    Parameters:
    initial_guess (float): Initial guess for the distance between atoms (default is 4 Å).
    epsilon (float): Depth of the potential well (default is 0.01 eV).
    sigma (float): Distance at which the potential is zero (default is 3.4 Å).
    
    Returns:
    OptimizeResult: The result of the optimization, including the optimal distance and the minimized potential energy.
    """

    result = minimize(lennard_jones, x0=initial_guess )
    return result
opt_ar = optimize_ar_dimer()
print(opt_ar)
#From this, the potential is minimized when r is 3.816, which is close to 4. The minimized potential is -0.00999999995038553 which is close to -0.01eV.


# %%
#ploting the Lennard-Jones potential V (r) as a function of the distance r between 3 Å ≤ r ≤ 6  Å.
distance = np.linspace(3,6,100)
distance_array = np.array([3,4,5,6])

plt.plot(distance_array, lennard_jones(distance_array),'o')
plt.plot(distance, lennard_jones(distance), label= 'Lenard-Jones Potential')
plt.xlabel('Distance (Å)')
plt.ylabel('Potential Energy (eV)')
plt.title('Lennard-Jones Potential')
plt.axvline(x= opt_ar.x[0], color = 'green', linestyle= '--', label= f'Equilibrium Distance ({opt_ar.x[0]:.2f} Å)')
plt.scatter(opt_ar.x[0], lennard_jones(opt_ar.x[0]), color='red', zorder=5, label='Equilibrium Point')
plt.legend()

plt.savefig('homework-2-1/argon_dimer_potential.png') 
plt.show()
# %%
#Creating xyz file for Argon Dimer 
def save_dimer_to_xyz(filename, r12):
    """
    Saves the atomic coordinates of the argon dimer to an XYZ file.

    Parameters:
    filename (str): The name of the file where the coordinates will be saved.
    r12 (float): The optimized bond length (distance) between the two argon atoms (in Å).
    
    Returns:
    None: Writes the atomic coordinates to an XYZ file in the standard format.
    """
    with open(filename, 'w') as f:
        f.write("2\nArgon Dimer\n")
        f.write(f"Ar 0.000000 0.000000 0.000000\n")
        f.write(f"Ar {r12:.6f} 0.000000 0.000000\n")


# %%

if __name__ == "__main__":
    optimal_r12 =opt_ar.x[0]
    print(f"Optimal bond length (r12): {optimal_r12:.6f} Å")
    save_dimer_to_xyz("homework-2-1/argon_dimer.xyz", optimal_r12)