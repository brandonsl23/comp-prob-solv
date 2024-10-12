#%%
import numpy as np 
from scipy.integrate import trapezoid  

#Isothermal process 

def isothermal_work (Vi,Vf, n, R, T):
    """
    Compute the work done during an isothermal expansion of an ideal gas.

    Parameters:
    Vi (float): Initial volume of the gas (in cubic meters, m³)
    Vf (float): Final volume of the gas (in cubic meters, m³)
    n (float): Number of moles of the ideal gas
    R (float): Ideal gas constant (J/mol·K)
    T (float): Initial temperature of the gas (in Kelvin)
    
    Returns:
    float: Work done on the gas during the adiabatic expansion (in Joules, J)
"""

    V = np.linspace(Vi,Vf,1000)
    P = n*R*T/V
    work = -trapezoid(P,V)
    return work 
# %%
#using the parameters from table 1 for ideal gas
## parameter | Value | units
n = 1  # moles of ideal gas
R = 8.314  # J/mol-K
T = 300  # Kelvin
Vi = 0.1  # Initial volume in m^3
Vf = 3 * Vi  # Final volume

#%%
# Compute work for isothermal expansion
work_isothermal = isothermal_work(Vi, Vf, n, R, T)
print(f"Work done in isothermal expansion: {work_isothermal} J")
# %%
