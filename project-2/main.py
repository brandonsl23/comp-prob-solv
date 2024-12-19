#%% Import Required Libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from PIL import Image
import csv
import io

#%% Simulation Parameters
k_B = 1.0  # Boltzmann constant (scaled)
dt = 0.01  # Time step
total_steps = 10000  # Number of steps
box_size = 100.0  # Size of the cubic box
mass = 1.0  # Particle mass
r0 = 1.0  # Equilibrium bond length
rescale_interval = 100  # Steps between velocity rescaling
n_particles = 20  # Number of particles
k = 1.0  # Spring constant
epsilon_repulsive = 0.5  # Depth of repulsive LJ potential
epsilon_attractive = 0.5  # Depth of attractive LJ potential
sigma = 1.0  # Lennard-Jones potential parameter
cutoff = 2 ** (1 / 6) * sigma  # LJ cutoff distance


# Output directory
output_dir = "Data"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#%% Helper Functions
def apply_pbc(position, box_size):
    return position % box_size

def minimum_image(displacement, box_size):
    return displacement - box_size * np.round(displacement / box_size)

def initialize_chain(n_particles, box_size, r0):
    positions = np.zeros((n_particles, 3))
    current_position = [box_size / 2, box_size / 2, box_size / 2]
    positions[0] = current_position
    for i in range(1, n_particles):
        direction = np.random.randn(3)
        direction /= np.linalg.norm(direction)
        next_position = current_position + r0 * direction
        positions[i] = apply_pbc(next_position, box_size)
        current_position = positions[i]
    return positions

def initialize_velocities(n_particles, target_temperature, mass):
    velocities = np.random.normal(0, np.sqrt(target_temperature / mass), (n_particles, 3))
    velocities -= np.mean(velocities, axis=0)
    return velocities

def compute_harmonic_forces(positions, k, r0, box_size):
    forces = np.zeros_like(positions)
    for i in range(0, n_particles - 1):
        displacement = positions[i + 1] - positions[i]
        displacement = minimum_image(displacement, box_size)
        distance = np.linalg.norm(displacement)
        force_magnitude = -k * (distance - r0)
        force = force_magnitude * (displacement / distance)
        forces[i] -= force
        forces[i + 1] += force
    return forces

def compute_lennard_jones_forces(positions, epsilon, sigma, box_size):
    forces = np.zeros_like(positions)
    for i in range(n_particles - 1):
        for j in range(i + 1, n_particles):
            displacement = positions[j] - positions[i]
            displacement = minimum_image(displacement, box_size)
            distance = np.linalg.norm(displacement)
            if distance < cutoff:
                force_magnitude = (24 * epsilon * ((2 * (sigma / distance) ** 12) - (sigma / distance) ** 6) / distance)
                force = force_magnitude * (displacement / distance)
                forces[i] -= force
                forces[j] += force
    return forces

def compute_forces(positions):
    lj_forces_repulsive = compute_lennard_jones_forces(positions, epsilon_repulsive, sigma, box_size)
    lj_forces_attractive = compute_lennard_jones_forces(positions, epsilon_attractive, sigma, box_size)
    harmonic_forces = compute_harmonic_forces(positions, k, r0, box_size)
    return harmonic_forces + lj_forces_repulsive + lj_forces_attractive

def compute_potential_energy(positions):
    potential_energy = 0
    for i in range(n_particles - 1):
        displacement = minimum_image(positions[i + 1] - positions[i], box_size)
        distance = np.linalg.norm(displacement)
        potential_energy += 0.5 * k * (distance - r0) ** 2

    for i in range(n_particles):
        for j in range(i + 1, n_particles):
            displacement = minimum_image(positions[i] - positions[j], box_size)
            distance = np.linalg.norm(displacement)
            if distance < cutoff:
                potential_energy += 4 * epsilon_repulsive * ((sigma / distance) ** 12 - (sigma / distance) ** 6 + 0.25)
                potential_energy += 4 * epsilon_attractive * ((sigma / distance) ** 12 - (sigma / distance) ** 6)
    return potential_energy

def velocity_verlet(positions, velocities, forces, dt, mass):
    velocities += 0.5 * forces / mass * dt
    positions += velocities * dt
    positions = apply_pbc(positions, box_size)
    forces_new = compute_forces(positions)
    velocities += 0.5 * forces_new / mass * dt
    return positions, velocities, forces_new

def rescale_velocities(velocities, target_temperature, mass):
    kinetic_energy = 0.5 * mass * np.sum(np.linalg.norm(velocities, axis=1) ** 2)
    current_temperature = (2 / 3) * kinetic_energy / (n_particles * k_B)
    scaling_factor = np.sqrt(target_temperature / current_temperature)
    velocities *= scaling_factor
    return velocities

def calculate_radius_of_gyration(positions):
    center_of_mass = np.mean(positions, axis=0)
    Rg_squared = np.mean(np.sum((positions - center_of_mass) ** 2, axis=1))
    return np.sqrt(Rg_squared)

def calculate_end_to_end_distance(positions):
    return np.linalg.norm(positions[-1] - positions[0])

def save_frame(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return Image.open(buf)

#%% Simulation Function with Visualization
def run_simulation_with_labeled_frames(target_temperature, n_particles, box_size, r0, mass, dt, total_steps, rescale_interval, output_filename):
    positions = initialize_chain(n_particles, box_size, r0)
    velocities = initialize_velocities(n_particles, target_temperature, mass)
    total_forces = compute_forces(positions)

    frames = []  # To store visualization frames

    for step in range(total_steps):
        positions, velocities, total_forces = velocity_verlet(positions, velocities, total_forces, dt, mass)

        if step % rescale_interval == 0:
            velocities = rescale_velocities(velocities, target_temperature, mass)

        if step % 100 == 0:  # Visualization every 100 steps
            fig = plt.figure(figsize=(6, 6))
            ax = fig.add_subplot(111, projection="3d")
            ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=50, color='blue')
            ax.text2D(0.05, 0.95, f"Temperature: {target_temperature:.2f}", transform=ax.transAxes, fontsize=12, color='red')
            ax.set_xlim(0, box_size)
            ax.set_ylim(0, box_size)
            ax.set_zlim(0, box_size)
            ax.set_title(f"Step {step}, T = {target_temperature:.2f}")
            frames.append(save_frame(fig))
            plt.close(fig)

    # Save GIF
    frames[0].save(output_filename, save_all=True, append_images=frames[1:], duration=200, loop=0)
    ase_atoms = Atoms('X' * n_particles, positions=positions)

  # Collect Rg and Ree at end of simulation
    Rg = (calculate_radius_of_gyration(positions))
    Ree = (calculate_end_to_end_distance(positions))  

    return ase_atoms, Rg, Ree


#%% Regular Simulation Function
def run_simulation(target_temperature, n_particles, box_size, r0, mass, dt, total_steps, rescale_interval):
    positions = initialize_chain(n_particles, box_size, r0)
    velocities = initialize_velocities(n_particles, target_temperature, mass)
    total_forces = compute_forces(positions)

    for step in range(total_steps):
        positions, velocities, total_forces = velocity_verlet(positions, velocities, total_forces, dt, mass)

        if step % rescale_interval == 0:
            velocities = rescale_velocities(velocities, target_temperature, mass)

        # Collect Rg and Ree at end of simulation
    Rg = (calculate_radius_of_gyration(positions))
    Ree = (calculate_end_to_end_distance(positions))

    return positions, velocities, total_forces, Rg, Ree


#%% Simulation for Single Temperature
target_temperature = 0.1
positions, velocities, total_forces, Rg, Ree = run_simulation(target_temperature, n_particles, box_size, r0, mass, dt, total_steps, rescale_interval)

# Analyze the Rg and Ree lists
print(f"Single Simulation:  Rg = {Rg:.3f}, Ree = {Ree:.3f}")


#%% Simulation over Temperature Range
Rg_values, Ree_values, potential_energies = [], [], []
temperatures = np.linspace(0.1, 1.0, 10)  # Temperature range

for T in temperatures:
    print(f"Running simulation for T = {T:.2f}")
    ase_atoms, Rg, Ree = run_simulation_with_labeled_frames(T, n_particles, box_size, r0, mass, dt, total_steps, rescale_interval, os.path.join(output_dir, f"polymer_T_{T:.2f}.gif"))
    Rg_values.append(Rg)
    Ree_values.append(Ree)
    potential_energies.append(compute_potential_energy(ase_atoms.positions))

# Save results to CSV
output_csv = os.path.join(output_dir, "temperature_results.csv")
with open(output_csv, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Temperature", "Radius_of_Gyration", "End_to_End_Distance", "Potential_Energy"])
    for T, Rg, Ree, PE in zip(temperatures, Rg_values, Ree_values, potential_energies):
        writer.writerow([T, Rg, Ree, PE])

print(f"Results saved to {output_csv}")

#%% Plot Results
plt.figure()
plt.plot(temperatures, Rg_values, label='Radius of Gyration')
plt.xlabel('Temperature')
plt.ylabel('Radius of Gyration')
plt.title('Radius of Gyration vs Temperature')
plt.legend()
rg_plot_path = os.path.join(output_dir, "radius_of_gyration_vs_temperature.png")
plt.savefig(rg_plot_path)
plt.show()

plt.figure()
plt.plot(temperatures, Ree_values, label='End-to-End Distance')
plt.xlabel('Temperature')
plt.ylabel('End-to-End Distance')
plt.title('End-to-End Distance vs Temperature')
plt.legend()
ree_plot_path = os.path.join(output_dir, "end_to_end_distance_vs_temperature.png")
plt.savefig(ree_plot_path)
plt.show()

plt.figure()
plt.plot(temperatures, potential_energies, label='Potential Energy')
plt.xlabel('Temperature')
plt.ylabel('Potential Energy')
plt.title('Potential Energy vs Temperature')
plt.legend()
pe_plot_path = os.path.join(output_dir, "potential_energy_vs_temperature.png")
plt.savefig(pe_plot_path)
plt.show()


# %%
# Parameter Ranges to Test
k_values = np.linspace(1.0, 3.0, 3)  # Spring constants in a narrower range
epsilon_repulsive_values = np.linspace(0.5, 2.0, 3)  # Smaller repulsive strengths
low_temperature = 0.1  # Low temperature for folding test

# Storage for Results
Rg_results = []
Ree_results = []

# Sweep through k and epsilon_repulsive
output_csv_k_eps = os.path.join(output_dir, "k_epsilon_results.csv")
with open(output_csv_k_eps, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["k", "epsilon_repulsive", "Rg", "Ree"])  # CSV Header

    for k_test in k_values:
        for epsilon_repulsive_test in epsilon_repulsive_values:
            # Update the parameters for this run
            k = k_test
            epsilon_repulsive = epsilon_repulsive_test

            # Run the simulation without visualization for efficiency
            positions, velocities, total_forces, Rg, Ree = run_simulation(
                low_temperature, n_particles, box_size, r0, mass, dt, total_steps, rescale_interval
            )
        
            # Store results
            Rg_results.append((k_test, epsilon_repulsive_test, Rg))
            Ree_results.append((k_test, epsilon_repulsive_test, Ree))

            # Write to CSV
            writer.writerow([k_test, epsilon_repulsive_test, Rg, Ree])

print(f"Results for k and epsilon_repulsive saved to {output_csv_k_eps}")





