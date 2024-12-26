import numpy as np
import scipy.io as sio

# Set parameters
n_points = 256  # Number of spatial points
n_timesteps = 100  # Number of time steps
T = 2.475  # Final time
x_min, x_max = 0, 2*np.pi  # Spatial domain
nu = 0.01  # Viscosity for Burgers' equation
num_samples = 5000  # Number of samples

# Create spatial and temporal grid
x = np.linspace(x_min, x_max, n_points)  # Spatial grid
t = np.linspace(0, T, n_timesteps)  # Temporal grid

# Create a meshgrid for the spatial and temporal coordinates
X, T_grid = np.meshgrid(x, t)

# Solve the Burgers' equation (or use a synthetic solution for simplicity)
# For synthetic data, we use a simple sinusoidal solution (a placeholder)
u = np.sin(np.pi * X) * np.exp(-nu * np.pi**2 * T_grid)

# Generate 5000 random samples
# Randomly select indices from the spatial grid (no time indices needed)
random_indices_x = np.random.randint(0, n_points, size=(num_samples, 1))
random_indices_t = np.random.randint(0, n_timesteps, size=(num_samples, 1))  # Random spatial indices

# Initialize arrays to store the data
x_data = np.zeros((num_samples, n_points))  # Spatial coordinates for each sample
y_data = np.zeros((num_samples, n_timesteps, n_points))  # Solution u(x, t) for each sample

# Generate data for each sample
for i in range(num_samples):
    # Select random spatial indices for this sample
    selected_space_indices = random_indices_x[i]
     # Randomly selected spatial points
    
    # Get the corresponding x values for this sample (spatial coordinates)
    x_data[i, :] = x[selected_space_indices]  # Spatial values for this sample
    
    # For each time step, get the solution u(x, t) at the selected spatial points
    for j in range(n_timesteps):
        y_data[i, j, :] = u[j, selected_space_indices]  # Solution for each time step at the selected x points

# Save the data to a .mat file
mat_data = {
    'x_data': x_data,  # [n_samples, n_points] (spatial coordinates)
    'y_data': y_data,  # [n_samples, n_timesteps, n_points] (solution u(x, t))
}

# Save to a .mat file
sio.savemat('burgers_data_samples.mat', mat_data)

print(f"5000 samples have been saved to 'burgers_data_samples.mat'")



# import numpy as np
# import matplotlib.pyplot as plt

# # Parameters
# L = 2*np.pi           # Length of the spatial domain (x ∈ [0, L])
# T = 2.475          # Total time
# Nx = 256          # Number of spatial points
# Nt = 100          # Number of time steps
# nu = 0.1 
# n_samples = 5000        # Kinematic viscosity (diffusivity)

# # Discretization
# dx = L / (Nx - 1)         # Spatial step size
# dt = T / Nt               # Time step size
# x = np.linspace(0, L, Nx) # Spatial grid
# t = np.linspace(0, T, Nt) # Time grid


# # Initial condition: a sine wave
# u0 = np.sin(2 * np.pi * x)  # Initial condition u(x, 0)

# # Initialize u as an array for storing solutions at each time step
# u = np.zeros((Nt, Nx))  # Array to store u values at each time step
# u[0, :] = u0           # Set the initial condition

# # Lax-Friedrichs scheme for solving Burgers' equation
# for n in range(0, Nt-1):
#     for i in range(1, Nx-1):
#         u[n+1, i] = 0.5 * (u[n, i+1] + u[n, i-1]) - 0.5 * (dt/dx) * (u[n, i+1]**2 - u[n, i-1]**2) + nu * (dt/dx**2) * (u[n, i+1] - 2*u[n, i] + u[n, i-1])
    
#     # Periodic boundary conditions
#     u[n+1, 0] = u[n+1, -2]
#     u[n+1, -1] = u[n+1, 1]

# # Plot some time snapshots
# plt.figure(figsize=(8, 6))
# for n in range(0, Nt, Nt // 5):  # Plot snapshots at different time steps
#     plt.plot(x, u[n, :], label=f't={n*dt:.2f}')

# plt.title('Burgers\' Equation Solution')
# plt.xlabel('x')
# plt.ylabel('u(x, t)')
# plt.legend()
# plt.grid(True)
# plt.show()

# # Optionally, save the dataset as (x, t, u(x, t)) pairs
# data = []
# for n in range(Nt):
#     for i in range(Nx):
#         data.append([x[i], t[n], u[n, i]])

# data = np.array(data)

# # Save to a .mat file (can be loaded into MATLAB)
# import scipy.io
# scipy.io.savemat('burgers_solution.mat', {'data': data, 'x': x, 't': t})