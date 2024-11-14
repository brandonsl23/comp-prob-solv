#%%
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.constants import k 

#%%
#%% parameters 
mus_A = np.linspace(-0.2, 0, 7)  # Array for chemical potential values of hydrogen
Ts = np.linspace(0.001, 0.019, 7)  # Array for temperature values
print(Ts)
# Convert Ts from energy units to kelvin by dividing by the Boltzmann constant (in eV/K)
k_eV = k * 6.242e+18  # Convert k from J/K to eV/K
Ts_kelvin = Ts / k_eV  # Convert temperatures to K

#creating parameter sets 
params_ideal = {
    'epsilon_A': -0.1,   # epsilon_N
    'epsilon_B': -0.1,   # epsilon_H
    'epsilon_AA': 0,     # epsilon_NN
    'epsilon_BB': 0,     # epsilon_HH
    'epsilon_AB': 0      # epsilon_NH
}

params_repulsive = {
    'epsilon_A': -0.1,   # epsilon_N
    'epsilon_B': -0.1,   # epsilon_H
    'epsilon_AA': 0.05,  # epsilon_NN
    'epsilon_BB': 0.05,  # epsilon_HH
    'epsilon_AB': 0.05   # epsilon_NH
}

params_attractive = {
    'epsilon_A': -0.1,     # epsilon_N
    'epsilon_B': -0.1,     # epsilon_H
    'epsilon_AA': -0.05,   # epsilon_NN
    'epsilon_BB': -0.05,   # epsilon_HH
    'epsilon_AB': -0.05    # epsilon_NH
}

params_immiscible = {
    'epsilon_A': -0.1,     # epsilon_N
    'epsilon_B': -0.1,     # epsilon_H
    'epsilon_AA': -0.05,   # epsilon_NN
    'epsilon_BB': -0.05,   # epsilon_HH
    'epsilon_AB': 0.05     # epsilon_NH
}

params_like_dissolves_unlike = {
    'epsilon_A': -0.1,    # epsilon_N
    'epsilon_B': -0.1,    # epsilon_H
    'epsilon_AA': 0.05,   # epsilon_NN
    'epsilon_BB': 0.05,   # epsilon_HH
    'epsilon_AB': -0.05   # epsilon_NH
}

# Organize all parameter sets into a dictionary
parameter_sets = {
    "Ideal Mixture": params_ideal,
    "Repulsive Interactions": params_repulsive,
    "Attractive Interactions": params_attractive,
    "Immiscible": params_immiscible,
    "Like Dissolves Unlike": params_like_dissolves_unlike
}
#%%
# Initialize Lattice
def initialize_lattice(size):
    """Create a 2D array 'lattice' of dimensions size x size, initialized to 0 (empty sites)."""
    lattice = np.zeros((size, size), dtype=int)
    return lattice

#%%
# Compute Neighbor Indices with Periodic Boundary Conditions
def compute_neighbor_indices(size):
    """Compute neighbor indices for each site in a 2D lattice with periodic boundary conditions."""
    neighbor_indices = {}
    for x in range(size):
        for y in range(size):
            neighbors = [
                ((x - 1) % size, y),    # Up
                ((x + 1) % size, y),    # Down
                (x, (y - 1) % size),    # Left
                (x, (y + 1) % size)     # Right
            ]
            neighbor_indices[(x, y)] = neighbors
    return neighbor_indices

#%%
# Calculate Interaction Energy
def calculate_interaction_energy(lattice, site, particle, neighbor_indices, epsilon_AA, epsilon_BB, epsilon_AB):
    """Calculate the interaction energy for a particle at a given site with its neighbors."""
    x, y = site
    interaction_energy = 0
    for neighbor in neighbor_indices[(x, y)]:
        neighbor_particle = lattice[neighbor]
        if neighbor_particle != 0:
            if particle == 1:  # Particle A
                if neighbor_particle == 1:
                    interaction_energy += epsilon_AA
                else:  # Neighbor is Particle B
                    interaction_energy += epsilon_AB
            else:  # Particle B
                if neighbor_particle == 2:
                    interaction_energy += epsilon_BB
                else:  # Neighbor is Particle A
                    interaction_energy += epsilon_AB
    return interaction_energy

#%%
# Attempt to Add or Remove a Particle
def attempt_move(lattice, N_A, N_B, N_empty, neighbor_indices, params):
    """Add or remove a particle from the lattice using the Metropolis algorithm."""
    size = lattice.shape[0]
    N_sites = size * size
    beta = 1 / (params['T'])
    epsilon_A, epsilon_B = params['epsilon_A'], params['epsilon_B']
    epsilon_AA, epsilon_BB, epsilon_AB = params['epsilon_AA'], params['epsilon_BB'], params['epsilon_AB']
    mu_A, mu_B = params['mu_A'], params['mu_B']
    
    # Decide whether to add or remove a particle (50% chance each)
    if random.choice(['add', 'remove']) == 'add':
        if N_empty == 0:
            return N_A, N_B, N_empty  # No empty sites available
        empty_sites = np.argwhere(lattice == 0)
        site = tuple(random.choice(empty_sites))
        
        # Decide which particle to add (A or B) with equal probability
        if random.choice(['A', 'B']) == 'A':
            particle = 1  # Particle A
            mu = mu_A
            epsilon = epsilon_A
            N_s = N_A
        else:
            particle = 2  # Particle B
            mu = mu_B
            epsilon = epsilon_B
            N_s = N_B
        
        # Calculate energy change and acceptance probability
        delta_E = epsilon + calculate_interaction_energy(lattice, site, particle, neighbor_indices, epsilon_AA, epsilon_BB, epsilon_AB)
        acc_prob = min(1, (N_empty / (N_s + 1)) * np.exp(-beta * (delta_E - mu)))
        
        if random.random() < acc_prob:
            lattice[site] = particle
            if particle == 1:
                N_A += 1
            else:
                N_B += 1
            N_empty -= 1
    else:
        if N_sites - N_empty == 0:
            return N_A, N_B, N_empty  # No particles to remove
        occupied_sites = np.argwhere(lattice > 0)
        site = tuple(random.choice(occupied_sites))
        particle = lattice[site]
        
        if particle == 1:
            mu = mu_A
            epsilon = epsilon_A
            N_s = N_A
        else:
            mu = mu_B
            epsilon = epsilon_B
            N_s = N_B
            
        delta_E = -epsilon - calculate_interaction_energy(lattice, site, particle, neighbor_indices, epsilon_AA, epsilon_BB, epsilon_AB)
        acc_prob = min(1, (N_s / (N_empty + 1)) * np.exp(-beta * (delta_E + mu)))
        
        if random.random() < acc_prob:
            lattice[site] = 0
            if particle == 1:
                N_A -= 1
            else:
                N_B -= 1
            N_empty += 1
            
    return N_A, N_B, N_empty

#%%
# Run the GCMC Simulation
def run_simulation(size, n_steps, params):
    """Run the GCMC simulation for a specified number of steps."""
    random.seed(42)
    lattice = initialize_lattice(size)
    neighbor_indices = compute_neighbor_indices(size)
    N_sites = size * size
    N_A, N_B, N_empty = 0, 0, N_sites
    coverage_A, coverage_B = np.zeros(n_steps), np.zeros(n_steps)

    for step in range(n_steps):
        N_A, N_B, N_empty = attempt_move(lattice, N_A, N_B, N_empty, neighbor_indices, params)
        coverage_A[step] = N_A / N_sites
        coverage_B[step] = N_B / N_sites

    return lattice, coverage_A, coverage_B

#%%
# Plot Lattice Configuration
def plot_lattice(lattice, ax, title):
    """Plot the lattice configuration with gridlines and labels."""
    size = lattice.shape[0]

    # Plot particles on the lattice
    for x in range(size):
        for y in range(size):
            if lattice[x, y] == 1:
                # Plot a red circle for Particle A at (x + 0.5, y + 0.5) on axis 'ax'
                ax.plot(y + 0.5, size - x - 0.5, 'ro', markersize=10)
            elif lattice[x, y] == 2:
                # Plot a blue circle for Particle B at (x + 0.5, y + 0.5) on axis 'ax'
                ax.plot(y + 0.5, size - x - 0.5, 'bo', markersize=10)

    # Set axis limits and labels
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_xticks(range(size + 1))
    ax.set_yticks(range(size + 1))
    ax.grid(True, which='both')  # Enable grid lines

    # Remove x and y tick labels
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    # Set the title
    ax.set_title(title)

    return ax
# %%
#example snipet to test code!
# Parameters
size = 4
n_steps = 10000
mus_A = np.linspace(-0.2, 0, 7)
Ts = np.linspace(0.001, 0.019, 7)
params = []
for mu_A in mus_A:
    for T in Ts:
        params.append({
            'epsilon_A': -0.1,
            'epsilon_B': -0.1,
            'epsilon_AA': 0,
            'epsilon_BB': 0,
            'epsilon_AB': 0,
            'mu_A': mu_A,
            'mu_B': -0.1,
            'T': T  # Temperature (in units of k)
        })

# Run the simulation

final_lattice = np.zeros((len(mus_A), len(Ts), size, size))
mean_coverage_A = np.zeros((len(mus_A), len(Ts)))
mean_coverage_B = np.zeros((len(mus_A), len(Ts)))
for i, param in enumerate(params):
    lattice, coverage_A, coverage_B = run_simulation(size, n_steps, param)
    final_lattice[i // len(Ts), i % len(Ts)] = lattice
    mean_coverage_A[i // len(Ts), i % len(Ts)] = np.mean(coverage_A[-1000:])
    mean_coverage_B[i // len(Ts), i % len(Ts)] = np.mean(coverage_B[-1000:])

# Plot the T-mu_A phase diagram
fig, axs = plt.subplot_mosaic([[0, 1, 2], [3, 4, 5]], figsize=(10, 5)) #changed the figsize compared to the code provided for better visual!

# Mean coverage of A
axs[0].pcolormesh(mus_A, Ts_kelvin, mean_coverage_A.T, cmap='viridis', vmin=0, vmax=1)
axs[0].set_title(r'$\langle \theta_A \rangle$')
axs[0].set_xlabel(r'$\mu_A$')
axs[0].set_ylabel(r'$T$')

# Mean coverage of B
axs[1].pcolormesh(mus_A, Ts_kelvin, mean_coverage_B.T, cmap='viridis', vmin=0, vmax=1)
axs[1].set_title(r'$\langle \theta_B \rangle$')
axs[1].set_xlabel(r'$\mu_A$')
axs[1].set_yticks([])

# Mean total coverage
cax = axs[2].pcolormesh(mus_A, Ts_kelvin, mean_coverage_A.T + mean_coverage_B.T, cmap='viridis', vmin=0, vmax=1)
axs[2].set_title(r'$\langle \theta_A + \theta_B \rangle$')
axs[2].set_xlabel(r'$\mu_A$')
axs[2].set_yticks([])
fig.colorbar(cax, ax=axs[2], location='right', fraction=0.1)

# Plot the final lattice configuration

# mu_A = -0.2 eV and T = 0.01 / k
axs[3] = plot_lattice(final_lattice[0, 3], axs[3], r'$\mu_A = -0.2$ eV, $T = {:.0f}$ K'.format(0.01 / k_eV) )
# mu_A = -0.1 eV and T = 0.01 / k
axs[4] = plot_lattice(final_lattice[3, 3], axs[4], r'$\mu_A = -0.1$ eV, $T = {:.0f}$ K'.format(0.01 / k_eV) )
# mu_A = 0 eV and T = 0.01 / k
axs[5] = plot_lattice(final_lattice[6, 3], axs[5], r'$\mu_A = 0$ eV, $T = {:.0f}$ K'.format(0.01 / k_eV) )

plt.tight_layout()
plt.show()

# %%
selected_T = Ts[3]  # Choose a temperature in energy units for consistency in snapshots!!

# Loop through each parameter set to generate individual phase diagrams
for name, params in parameter_sets.items():
    mean_coverage_A = np.zeros((len(mus_A), len(Ts)))
    mean_coverage_B = np.zeros((len(mus_A), len(Ts)))

    # Setting up the figure for plotting phase diagrams for the current parameter set
    fig, axs = plt.subplot_mosaic([[0, 1, 2], [3, 4, 5]], figsize=(10, 5))
    fig.suptitle(f"Phase Diagram for {name}", fontsize=16) 
    print(f"Generating phase diagram for {name}...")

    # Run simulations across the parameter space for the current parameter set
    for i, mu_H in enumerate(mus_A):
        for j, T in enumerate(Ts):
            # Update the parameter set with current mu_A and T
            params['mu_A'] = mu_H
            params['T'] = T
            params['mu_B'] = -0.1

            lattice, coverage_A, coverage_B = run_simulation(size=4, n_steps=10000, params=params)
            mean_coverage_A[i, j] = np.mean(coverage_A[-1000:])
            mean_coverage_B[i, j] = np.mean(coverage_B[-1000:])

            # Plot lattice configurations for selected mu_A and selected_T values
            if T == selected_T and mu_H in [-0.2, -0.1, 0]:  # Adjust mu_H values as needed
                plot_index = 3 + [-0.2, -0.1, 0].index(mu_H)
                axs[plot_index] = plot_lattice(
                    lattice, axs[plot_index], f'$\mu_A = {mu_H:.1f}$ eV, $T = {T/k_eV:.0f}$ K'
                )

    # Plot the nitrogen coverage (⟨θ_A⟩)
    c1 = axs[0].pcolormesh(mus_A, Ts / k_eV, mean_coverage_A.T, cmap='viridis', shading='auto')
    axs[0].set_title("Nitrogen Coverage (θ_A)")
    axs[0].set_xlabel(r"$\mu_H$")
    axs[0].set_ylabel(r"$T$ (K)")
    fig.colorbar(c1, ax=axs[0], label="Coverage")

    # Plot the hydrogen coverage (⟨θ_B⟩)
    c2 = axs[1].pcolormesh(mus_A, Ts / k_eV, mean_coverage_B.T, cmap='viridis', shading='auto')
    axs[1].set_title("Hydrogen Coverage (θ_B)")
    axs[1].set_xlabel(r"$\mu_H$")
    fig.colorbar(c2, ax=axs[1], label="Coverage")

    # Plot the total coverage (⟨θ_A + θ_B⟩)
    c3 = axs[2].pcolormesh(mus_A, Ts / k_eV, (mean_coverage_A + mean_coverage_B).T, cmap='viridis', shading='auto')
    axs[2].set_title("Total Coverage (θ_A + θ_B)")
    axs[2].set_xlabel(r"$\mu_H$")
    fig.colorbar(c3, ax=axs[2], label="Coverage")

    # Adjust layout and save the phase diagram for the current parameter set
    plt.tight_layout()
    plt.savefig(f"graphs/phase_diagram_{name.replace(' ', '_').lower()}.png")
    plt.show()

    print(f"Phase diagram for {name} saved as 'graphs/phase_diagram_{name.replace(' ', '_').lower()}.png'.")

# %%
