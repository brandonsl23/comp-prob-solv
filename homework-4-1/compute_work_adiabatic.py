#%%
import numpy as np
from scipy.integrate import trapezoid
#%%
def adiabatic_work (Vi, Vf, n, R, T, gamma): 
    """
    Compute the work done during an adiabatic expansion of an ideal gas.

    Parameters:
    Vi (float): Initial volume of the gas (in cubic meters, m³)
    Vf (float): Final volume of the gas (in cubic meters, m³)
    n (float): Number of moles of the ideal gas
    R (float): Ideal gas constant (J/mol·K)
    T (float): Initial temperature of the gas (in Kelvin)
    gamma (float): Adiabatic index (ratio of specific heats, C_P/C_V)
    
    Returns:
    float: Work done on the gas during the adiabatic expansion (in Joules, J)
"""

    Pi = n*R*T/Vi
    constant = Pi *Vi * gamma
    V = np.linspace(Vi, Vf, 1000)
    P = constant/ V**gamma
    work = -trapezoid(P,V)
    return work 
#%%
#using the parameters from table 1 for ideal gas
## parameter | Value | units
n = 1  # moles of ideal gas
R = 8.314  # J/mol-K
T = 300  # Kelvin
Vi = 0.1  # Initial volume in m^3
Vf = 3 * Vi  # Final volume
gamma = 1.4  # Adiabatic index

#%%
# Compute work for adiabatic expansion
work_adiabatic = adiabatic_work(Vi, Vf, n, R, T, gamma)
print(f"Work done in adiabatic expansion: {work_adiabatic} J")





# %%
