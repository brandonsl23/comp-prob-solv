# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# %%
# Collect the Cartesian coordinates of H2, H2O, and benzene from the CCCBDB website
#H2
Hydrogen ={
    "H1"  : [0.0000,0.0000,	0.0000], 
    "H2" : [0.0000,	0.0000,	0.7414]
}
Water = {
    "O1":	[0.0000, 0.0000, 0.1173],
    "H2":	[0.0000, 0.7572, -0.4692],
    "H3":	[0.0000, -0.7572,-0.4692]
}
Benzene = {
    "C1":[0.0000 ,1.3970, 0.0000],
    "C2": [1.2098, 0.6985, 0.0000],
    "C3": [1.2098, -0.6985, 0.0000],
    "C4": [0.0000, -1.3970, 0.0000],
    "C5": [-1.2098, -0.6985, 0.0000],
    "C6": [-1.2098,	0.6985, 0.0000],
    "H7": [0.0000, 2.4810, 0.0000],
    "H8": [2.1486, 1.2405, 0.0000],
    "H9": [2.1486, -1.2405, 0.0000],
    "H10": [0.0000, -2.4810, 0.0000],
    "H11": [-2.1486, -1.2405, 0.0000],
    "H12": [-2.1486, 1.2405, 0.0000]
}

print("Cartesian coordinates of Hydrogen: ", Hydrogen) 
print()
print("Cartesian coordinates of Water: ", Water)
print()
print("Cartesian coordinates of Benzene: ", Benzene)

# %%
def compute_bond_length(coord1, coord2):
    """
    Computes bond length given two lists that contain x,y, and z coordinates.

    Parameters:
    coord1 (list): list containing x,y, and z coordinates for an atom 
    coord2 (list): list containing x,y, and z coordinates for an atom 

    Returns:
    float: bond length in Angstroms
    """
    x1, y1, z1 = coord1
    x2, y2, z2 = coord2
    d = np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
    if d > 2: 
        print("WARNING: The length between these two atoms is too long")
    print(f"The bond length between {coord1} and {coord2} is {d} Å")
    return d

# %%
##This is just to test!
compute_bond_length(Benzene["C1"], Benzene["C2"])

# %%
type(compute_bond_length(Benzene["C1"], Benzene["C2"]))

# %%
def compute_bond_angle(coord1, coord2, coord3):
    """
    Computes bond angle given 3 lists that each contain x,y, and z coordinates.

    Parameters:
    coord1 (list): list containing x,y, and z coordinates for the first atom 
    coord2 (list): list containing x,y, and z coordinates for the second atom
    coord3 (list): list containing x,y, and z coordinates for the third atom

    Returns:
    float: bond angle in degrees 
    """
    A = np.array(coord1)
    B = np.array(coord2)
    C = np.array(coord3)

    #findng vectors
    AB = A-B
    BC = C-B
    #magnitudes of vectors 
    AB_mag = np.linalg.norm(AB)
    BC_mag = np.linalg.norm(BC)

    #finding dot products 
    dot_product = np.dot(AB,BC)
    cos_theta = dot_product /(AB_mag*BC_mag)
    angle_degrees = np.degrees(np.arccos(cos_theta))

    if angle_degrees < 90:
        classification = "Acute"
    elif angle_degrees == 90:
        classification = "Right"
    else:
        classification = "Obtuse"

    print(f"The bond angle between {coord1}, {coord2}, {coord3} is {angle_degrees} degrees, classified as {classification}.") #print type of angle
    
    return angle_degrees

# %%
#Testt!
compute_bond_angle(Water["O1"],Water["H2"],Water["H3"])

# %%
type(compute_bond_angle(Water["O1"],Water["H2"],Water["H3"]))

# %%
def calculate_all_bond_lengths(molecule):
    """
    Computes all of the bond lengths given a molcule .

    Parameters:
    molecule (dic): Dictionary where keys are atom labels and values are lists of Cartesian coordinates (x,y,z) 
    
    Returns:
    list: contains atom pair and it's bond length
    """
    bond_lengths = []
    atoms = list(molecule.keys())
    for atom1 in molecule: 
        coord1 = molecule[atom1]
        for atom2 in molecule:
            if atom1 != atom2 and atom1 < atom2:  
                coord2 = molecule[atom2]
                bond_length = compute_bond_length(coord1, coord2)
                bond_lengths.append((atom1, atom2, bond_length))
               
    print()#this is just for spacing as compute_bond_length returns some strings!
    for bond in bond_lengths:
        atom1, atom2, length = bond
        print(f"Bond length between {atom1} and {atom2}: {length} Å")  
    
    return bond_lengths

# %%
#Test!!
calculate_all_bond_lengths(Water)

# %%
type(calculate_all_bond_lengths(Water))

# %%
def calculate_all_bond_angles(molecule):
    """
    Computes all of the bond angles given a molecule .

    Parameters:
    molecule (dic): Dictionary where keys are atom labels and values are lists of Cartesian coordinates (x,y,z) 
    
    Returns:
    list: Contains atom pair and it's bond length
    """
    bond_angles = []
    
    atoms = list(molecule.keys())
    
    for i in range(len(atoms)):
        for j in range(len(atoms)):
            if i == j:
                continue
            for k in range(len(atoms)):
                if i == k or j == k:
                    continue
                atom1 = atoms[i]
                atom2 = atoms[j]  
                atom3 = atoms[k]
                
                coord1 = molecule[atom1]
                coord2 = molecule[atom2]
                coord3 = molecule[atom3]
                
                bond_angle = compute_bond_angle(coord1, coord2, coord3)
                bond_angles.append((atom1, atom2, atom3, bond_angle))
    
    print() #this is just for spacing as compute_bond_angle returns some strings!
    for angle in bond_angles:
        atom1, atom2, atom3, bond_angle = angle
        print(f"The bond angle between {atom1}, {atom2}, and {atom3} is {bond_angle}°")
    
    return bond_angles

# %%
calculate_all_bond_angles(Water)

# %%
type(calculate_all_bond_angles(Water))


