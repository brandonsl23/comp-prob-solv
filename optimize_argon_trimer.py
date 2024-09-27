# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from homework_1_2 import compute_bond_length
from homework_1_2 import compute_bond_angle
from optimize_argon_dimer import lennard_jones

# %%
def total_potential_trimer(vari, epsilon=0.01, sigma=3.4):
    """
    Computes the total potential energy for an argon trimer using the Lennard-Jones potential.

    Parameters:
    vari (list): A list containing three values [r12, x3, y3], where:
    r12 (float): Distance between atom 1 and atom 2.
    x3 (float): x-coordinate of atom 3.
    y3 (float): y-coordinate of atom 3.
    epsilon (float, optional): Depth of the potential well (default is 0.01 eV).
    sigma (float, optional): Distance at which the potential is zero (default is 3.4 Å).

    Returns:
    total_Vr (float): The total potential energy of the trimer, which is the sum of the Lennard-Jones interactions between atom pairs.
    """
    r12 , x3, y3 = vari 
    atom_1 = [0,0,0]
    atom_3 = [x3,y3,0]
    atom_2 = [r12,0,0]

    r13 = compute_bond_length(atom_1,atom_3) #this is the distance between atom 1 and 3
    r23 = compute_bond_length(atom_2,atom_3) #this is the distance between atom 2 and 3

    V12 = lennard_jones(r12, epsilon, sigma)
    V13 = lennard_jones(r13, epsilon, sigma)
    V23 = lennard_jones(r23, epsilon, sigma)
    
    total_Vr = V12 + V13 + V23 
    
    return total_Vr
# %%
#optimizing the trimer 
inital_guesses = [4,4,4]
opt_trimer = minimize (total_potential_trimer,x0 =inital_guesses) 
print(opt_trimer )

#The function V(r) is minimized to -0.03 eV when r12 = 3.816, x3 = 1.908, y3 = 3.305. 


#%%

#extracting r12, x3,and y3

r12_opt = opt_trimer.x[0]
x3_opt = opt_trimer.x[1]
y3_opt = opt_trimer.x[2]

#atom coordinates 
atom_1 = [0,0,0]
atom_3 = [x3_opt,y3_opt,0]
atom_2 = [r12_opt,0,0]

#recalculating the distances between atoms (the first calculation was in the function!)
r13_opt = compute_bond_length(atom_1,atom_3) #this is the distance between atom 1 and 3
r23_opt = compute_bond_length(atom_2,atom_3) #this is the distance between atom 2 and 3

#calculating the angles between atoms 
angle_123 = compute_bond_angle(atom_1,atom_2,atom_3)  # Angle at atom 2
angle_231 = compute_bond_angle(atom_1,atom_3,atom_2)  # Angle at atom 3
angle_312 = compute_bond_angle(atom_3,atom_1,atom_2)   # Angle at atom 1



# %%
print("Optimal Distances (in Å):")
print(f"r12 (Distance between atom 1 and 2): {r12_opt:.2f} Å")
print(f"r13 (Distance between atom 1 and 3): {r13_opt:.2f} Å")
print(f"r23 (Distance between atom 2 and 3): {r23_opt:.2f} Å")

print("\nOptimal Angles (in degrees):")
print(f"Angle 123 (at atom 2): {angle_123:.2f}°")
print(f"Angle 231 (at atom 3): {angle_231:.2f}°")
print(f"Angle 312 (at atom 1): {angle_312:.2f}°")


#The argon trimer forms a perfect equilateral triangle, with each side measuring 3.82 Å 
#and all internal angles precisely 60.00°. This symmetrical arrangement
#minimizes the potential energy, resulting in a stable configuration.

#%%
# def to save trimer as an xyz file 
def save_trimer_to_xyz(filename, atom_1, atom_2, atom_3):
    """
    Saves the atomic coordinates of the argon trimer in the XYZ file format.

    Parameters:
    filename (str): The name of the file where the coordinates will be saved.
    atom_1 (list): A list of [x, y, z] coordinates for atom 1.
    atom_2 (list): A list of [x, y, z] coordinates for atom 2.
    atom_3 (list): A list of [x, y, z] coordinates for atom 3.

    Returns:
    None: Writes the atomic coordinates to an XYZ file.
    """
    with open(filename, 'w') as f:
        f.write("3\nArgon Trimer\n")
        f.write(f"Ar {atom_1[0]:.6f} {atom_1[1]:.6f} {atom_1[2]:.6f}\n")
        f.write(f"Ar {atom_2[0]:.6f} {atom_2[1]:.6f} {atom_2[2]:.6f}\n")
        f.write(f"Ar {atom_3[0]:.6f} {atom_3[1]:.6f} {atom_3[2]:.6f}\n")

# %%
#saving trimer as an xyz file!
if __name__ == "__main__":
    save_trimer_to_xyz("homework-2-1/argon_trimer.xyz", atom_1, atom_2, atom_3)
