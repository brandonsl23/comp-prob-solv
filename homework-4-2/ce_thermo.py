#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import k
import pandas as pd

#%% 
#changing from eV to J
eV_to_J = 1.60218e-19 
#temp rang in K
T_range = np.linspace(300,2000,100)
# %%
def save_work_to_csv(T_range, U_values, F_values, S_values, filename):
    """
    Saves the computed thermodynamic properties (internal energy, free energy, and entropy)
    as a function of temperature to a CSV file.

    Parameters:
    - T_range (list or array): List or array of temperature values (K).
    - U_values (list of arrays): List containing arrays of internal energy values (J) for 
      each case (Isolated Ce3+, Ce3+ with SOC, and Ce3+ with SOC & CFS).
    - F_values (list of arrays): List containing arrays of free energy values (J) for 
      each case.
    - S_values (list of arrays): List containing arrays of entropy values (in Joules per Kelvin) for 
      each case.
    - filename (string): Name of the CSV file where the data will be saved.
    
    This function organizes the thermodynamic data into a Pandas DataFrame and saves it to a CSV file.
    """
    # dictionary to organize the data!
    data = {
        'Temperature (K)': T_range,
        'U Isolated (J)': U_values[0],
        'U SOC (J)': U_values[1],
        'U SOC & CFS (J)': U_values[2],
        'F Isolated (J)': F_values[0],
        'F SOC (J)': F_values[1],
        'F SOC & CFS (J)': F_values[2],
        'S Isolated (J/K)': S_values[0],
        'S SOC (J/K)': S_values[1],
        'S SOC & CFS (J/K)': S_values[2],
    }
    # Create a DataFrame
    df = pd.DataFrame(data) 
    # Save the DataFrame to a CSV file
    df.to_csv(filename, index=False)
    return

def partition_function_isolated_ce3 (T):
    """
    Computes the partition function for an isolated Ce3+ ion, assuming 14-fold degeneracy
    and all states having zero energy.

    Parameters:
    - T (float): Temperature (K).
    
    Returns:
    - float: Partition function value (degeneracy) for the isolated Ce3+ case.
    """
    g = 14 #this is the degeneracy state
    #all states have zero energy, thus the e of the partition function equal 1, just g 
    return g


def partition_function_Ce_with_SOC (T):
    """
    Computes the partition function for Ce3+ with spin-orbit coupling (SOC).
    The 14-fold degenerate states split into two groups: 2F5/2 (6 states) and 2F7/2 (8 states),
    with an energy difference of 0.28 eV between them.

    Parameters:
    - T (float): Temperature (in Kelvin).
    
    Returns:
    - float: Partition function value for the Ce3+ ion with SOC.
    """
    g1, g2 = 6,8
    delta_E = 0.28* eV_to_J
    Z = g1 + g2 * np.exp(-delta_E / (k * T))
    return Z

def partition_function_Ce_with_SOC_and_CFS (T):
    g_values = [4, 2, 4, 2, 2]
    E_values = [0, 0.12 * eV_to_J, 0.13 * eV_to_J, 0.07 * eV_to_J, 0.14 * eV_to_J]
    Z = sum(g * np.exp(-E / (k * T)) for g, E in zip(g_values, E_values))
    return Z

#%%
def compute_thermodynamic_properties(T_range, partition_function):
    """
    Computes the partition function for Ce3+ with both spin-orbit coupling (SOC)
    and crystal field splitting (CFS).
    
    The SOC-split levels (2F5/2 and 2F7/2) are further split by crystal field effects into
    five distinct energy levels, each with its own degeneracy and energy.

    Parameters:
    - T (float): Temperature (in Kelvin).
    
    Returns:
    - float: Partition function value for the Ce3+ ion with SOC and CFS.
    """
    Z_values = []
    for T in T_range:
        Z_values.append(partition_function(T))

    Z_values = np.array(Z_values)
    lnZ_values = np.log(Z_values)

    # Compute U, F, S across the entire temperature range
    U = -np.gradient(lnZ_values, T_range) * k * T_range**2  # d(lnZ)/dT multiplied by kT^2
    F = -k * T_range * lnZ_values  # F = -kBT lnZ , Helmholtz free energy
    S = -np.gradient(F, T_range)  # S = - ∂F/∂T, entropy 

    return U, F, S

#%%
def plot_thermodynamic_property(T_range, U_values, F_values, S_values, labels):
    """
    Computes the thermodynamic properties (internal energy, free energy, and entropy)
    across a temperature range using the partition function.

    Parameters:
    - T_range (list or array): List or array of temperature values (K).
    - partition_function (function): Function that computes the partition function for a given temperature.
    
    Returns:
    - U (array): Internal energy (J) for each temperature.
    - F (array): Free energy (in Joules) for each temperature.
    - S (array): Entropy (in Joules per Kelvin) for each temperature.
    
    The internal energy (U) is computed as -∂lnZ/∂β, free energy (F) as -k_B T lnZ, and entropy (S) as -∂F/∂T.
    """
    # Plot for Internal Energy
    plt.figure()
    for U, label in zip(U_values, labels):
        plt.plot(T_range, U, label=f'{label}')
    plt.xlabel(r'Temperature (K)')
    plt.ylabel(r'Internal Energy, $U$ (J)')
    plt.title(r'Internal Energy vs. Temperature')
    plt.legend()
    plt.grid(True)
    plt.savefig('homework-4-2/U_vs_T')
    plt.show()
    
    # Plot for Free Energy
    plt.figure()
    for F, label in zip(F_values, labels):
        plt.plot(T_range, F, label=f'{label}')
    plt.xlabel(r'Temperature (K)')
    plt.ylabel(r'Free Energy, $F$ (J)')
    plt.title(r'Free Energy vs. Temperature')
    plt.legend()
    plt.grid(True)
    
    plt.savefig('homework-4-2/F_vs_T')
    plt.show()
    # Plot for Entropy
    plt.figure()
    for S, label in zip(S_values, labels):
        plt.plot(T_range, S, label=f'{label}')
    plt.xlabel(r'Temperature (K)')
    plt.ylabel(r'Entropy, $S$ (J/K)')
    plt.title(r'Entropy vs. Temperature')
    plt.legend()
    plt.grid(True)
    plt.savefig('homework-4-2/S_vs_T')
    plt.show()
    return

# Compute thermodynamic properties for each case
U_isolated, F_isolated, S_isolated = compute_thermodynamic_properties(T_range, partition_function_isolated_ce3)
U_soc, F_soc, S_soc = compute_thermodynamic_properties(T_range, partition_function_Ce_with_SOC)
U_soc_cfs, F_soc_cfs, S_soc_cfs = compute_thermodynamic_properties(T_range, partition_function_Ce_with_SOC_and_CFS)

# Storing the computed U, F, S arrays in the respective lists
U_values = [U_isolated, U_soc, U_soc_cfs]
F_values = [F_isolated, F_soc, F_soc_cfs]
S_values = [S_isolated, S_soc, S_soc_cfs]
labels = [r'$Isolated$ $Ce^{3+}$', r'$Ce^{3+}$ $with$ $SOC$',  r'$Ce^{3+}$ $with$ $SOC$ $&$ $CFS$']

# Now, you can plot the results for internal energy, free energy, and entropy
plot_thermodynamic_property(T_range, U_values, F_values, S_values, labels)


# %%
#saving data to csv file
save_work_to_csv(T_range, U_values, F_values, S_values, 'homework-4-2/U_F_S_vs_T.csv')

# %%
