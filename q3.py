import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from mpl_toolkits.mplot3d import Axes3D



def loss_for_grid(alpha, sigma, df, theta=0.32):
    h_pred = df.apply(
        lambda row: solve_h(row["t"], row["c_y"], alpha, sigma, theta),
        axis=1
    )
    return ((df["h_obs"] - h_pred) ** 2).sum()

# ------------IMPORTING AND CLEANING THE DATA--------------------------

#Importing Prescott data 
df_calib = pd.read_csv('q2data/data_prescott.csv')

#Fixes all missing values in the data set
df_calib["period"] = df_calib["period"].ffill()
df_calib["country"] = df_calib["country"].ffill()

#Renaming columns so that they match my columns
df_calib = df_calib.rename(columns = {
    "h": "h_obs",
    "tau": "t",
    "c2y": "c_y"})

#Renaming columns so that they have full country names rather than abreviations
country_map = {
    "DEU": "Germany",
    "FRA": "France",
    "ITA": "Italy",
    "CAN": "Canada",
    "GBR": "United Kingdom",
    "JPN": "Japan",
    "USA": "United States"}
df_calib["country"] = df_calib["country"].replace(country_map)


# --------------------Q3A-------------------------------

#FOC that characterises the optimal hours worked which is non-linear
def foc_equilibrium_condition(h, tau, c_y, alpha, sigma, theta = 0.32):
    return alpha * h - (((1 - tau) * (1 - theta)) / c_y) * (100 - h) ** sigma

# Test values
tau = 0.44
c_y = 0.83
alpha = 1.7105
sigma = 1.0
theta = 0.32

print("FOC at h=1:", foc_equilibrium_condition(1, tau, c_y, alpha, sigma, theta))
print("FOC at h=21.1:", foc_equilibrium_condition(21.1, tau, c_y, alpha, sigma, theta))
print("FOC at h=99:", foc_equilibrium_condition(99, tau, c_y, alpha, sigma, theta))






def solve_h(tau, c_y, alpha, sigma, theta = 0.32):
    result = minimize_scalar(
        lambda h: foc_equilibrium_condition(h, tau, c_y, alpha, sigma, theta) ** 2,
        bounds=(1e-6, 99.999),
        method="bounded")

    return result.x

def solve_h(tau, c_y, alpha, sigma, theta=0.32):
    return brentq(foc_equilibrium_condition,1e-6,99.999,args=(tau, c_y, alpha, sigma, theta))


#Creating the loss function for alpha to be calibrated on
def loss_function(parameters, df, theta = 0.32):
    alpha, sigma = parameters

    #Placing a large penalty for invalid values (negatives)
    if alpha <= 0 or sigma <= 0 or alpha > 100000 or sigma > 50:
        return 1e10
    
    h_pred = df.apply(
        lambda row: solve_h(row["t"], row["c_y"], alpha, sigma, theta),
        axis = 1)
    loss = ((df["h_obs"] - h_pred) ** 2).sum()
    return float(loss)


res_lbfgs = minimize(
    loss_function,
    x0=[1.7, 1.0], # initial guess of alpha and sigma
    args=(df_calib,),
    method="L-BFGS-B",
    bounds=[(0.01, 100000), (0.01, 50)])

alpha_hat, sigma_hat = res_lbfgs.x

print("Bounded Results")
print(f"\nAlpha: {alpha_hat:.4f}")
print(f"Sigma: {sigma_hat:.4f}")
print(f"Loss: {res_lbfgs.fun:.4f}")
print("Success:", res_lbfgs.success)
print("Message:", res_lbfgs.message)

res_nm = minimize(
        loss_function,
        x0=[5000, 2.0], #This is the initial guess
        args = (df_calib,),
        method="Nelder-Mead")

alpha_hat, sigma_hat = res_nm.x

print("Nelder Mead Results:")
print(f"\nAlpha: {alpha_hat:.4f}")
print(f"Sigma: {sigma_hat:.4f}")
print(f"Loss: {res_nm.fun:.4f}")

# 1. Define the grid range
# We'll center the grid around your optimized results
alpha_range = np.linspace(alpha_hat * 0.5, alpha_hat * 1.5, 20)
sigma_range = np.linspace(sigma_hat * 0.5, sigma_hat * 1.5, 20)

# Create the meshgrid
A, S = np.meshgrid(alpha_range, sigma_range)
L = np.zeros(A.shape)

print("Generating loss surface... this may take a moment.")

# 2. Evaluate the loss function over the grid
for i in range(len(sigma_range)):
    for j in range(len(alpha_range)):
        # We call your existing loss_function
        L[i, j] = loss_function([A[i, j], S[i, j]], df_calib)

# 3. Plotting
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the surface
surf = ax.plot_surface(A, S, L, cmap='viridis', alpha=0.8, edgecolor='none')

# 4. Highlight the Global Minimum
ax.scatter(alpha_hat, sigma_hat, res_nm.fun, color='red', s=100, label='Global Minimum', zorder=5)

# Formatting
ax.set_xlabel(r'$\alpha$ (Alpha)')
ax.set_ylabel(r'$\sigma$ (Sigma)')
ax.set_zlabel('Loss (SSR)')
ax.set_title('3D Loss Function Surface')
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.legend()

plt.show()



