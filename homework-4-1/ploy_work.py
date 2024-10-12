#%%
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from  compute_work_adiabatic import adiabatic_work
from compute_work_isothermal import isothermal_work


#%% 
# Conversion of Work vs final volume data to a CSV file
def save_work_to_csv(volumes, work_isothermal, work_adiabatic, filename):
    """
    Saves the computed work data for isothermal and adiabatic processes to a CSV file.

    Parameters:
    - volumes (list or array): List or array of final volumes (m^3) for the gas expansion.
    - work_isothermal (list or array): List or array of work values (J) for the isothermal process corresponding to the final volumes.
    - work_adiabatic (list or array): List or array of work values (J) for the adiabatic process corresponding to the final volumes.
    - filename (string): Name of the CSV file where the data will be saved.
    
    This function organizes the final volumes, work done in the isothermal process, and work done in the adiabatic process into a Pandas DataFrame and saves the data into a CSV file.
    """
    # dictionary to organize the data!
    data = {
        'Final Volume (m^3)': volumes,
        'Work Isothermal (J)': work_isothermal,
        'Work Adiabatic (J)': work_adiabatic
    }
    
    df = pd.DataFrame(data)
    
    # DataFrame to a CSV file
    df.to_csv(filename, index=False)
    return
#%% 
def plot_work_vs_volume(volumes, work_isothermal, work_adiabatic):
    """
    Plots the work done vs. final volume for both isothermal and adiabatic expansion processes.

    Parameters:
    - volumes (list or array): List or array of final volumes (m^3) for the gas expansion.
    - work_isothermal (list or array): List or array of work values (in Joules) for the isothermal process corresponding to the final volumes.
    - work_adiabatic (list or array): List or array of work values (in Joules) for the adiabatic process corresponding to the final volumes.
    
    This function creates a plot comparing the work done in isothermal and adiabatic expansions as a function of final volume. 
    The plot is saved as a PNG image and is displayed on the screen.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(volumes, work_isothermal, label='Isothermal Process', marker='o')
    plt.plot(volumes, work_adiabatic, label='Adiabatic Process', marker='x')
    plt.xlabel('$Final$ $Volume$ $(m^3)$')
    plt.ylabel('$Work$ $(J)$')
    plt.title('$Work$ $Done$ $in$ $Isothermal$ $vs$ $Adiabatic$ $Expansion$')
    plt.legend()
    plt.grid(True)
    plt.savefig('homework-4-1/work_vs_volume.png')
    plt.show()
    return
#%%
#Constants 
n = 1  # moles of ideal gas
R = 8.314  # J/mol-K
T = 300  # Kelvin
Vi = 0.1  # Initial volume in m^3
gamma = 1.4  # Adiabatic index

final_volumes = np.linspace(Vi, 3 * Vi, 20)

work_iso_list = [isothermal_work(Vi, Vf, n, R, T) for Vf in final_volumes]
work_adi_list = [adiabatic_work(Vi, Vf, n, R, T, gamma)for Vf in final_volumes]

# Saving results to CSV
save_work_to_csv(final_volumes, work_iso_list, work_adi_list, 'homework-4-1/work_vs_volume.csv')

# Generate the plot!
plot_work_vs_volume(final_volumes, work_iso_list, work_adi_list)



