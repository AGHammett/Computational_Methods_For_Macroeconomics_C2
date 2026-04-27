import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import root_scalar
from dataclasses import dataclass

@dataclass(frozen=True)
class Params:
    alpha: float = 0.3
    beta: float = 0.96
    delta: float = 0.1
    k1: float = 1.0
    T: int = 30

def create_z_vector(T: int): 
    """
    Creates a vector of length T where T[i] is z_i
    """
    z_singles = (1 / 2.9) ** np.arange(T) # calculate the values for the ith term of the summation
    return np.cumsum(z_singles) # cumulatively sum each element for z_i

def k_steady_state(z_star, p: Params):
    """
    Returns k* for given set of parameters
    """
    rho = (1 / p.beta) - 1
    return ((rho + p.delta) / (z_star * p.alpha)) ** (1 / (p.alpha - 1))

def f(z, k, alpha): return z * k ** alpha # production function

def deriv_f(z, k, alpha): return alpha * z * k ** (alpha - 1) # derivative of production function wrt k

def k_T(k2: float, Z: np.array, p: Params):
    """
    Uses Euler equation to solve for k_{t+2} recursively up to k_T.
    Returns vector of k values
    """

    # create k vector and fill in initial values
    K = np.empty(p.T)
    K[0]= p.k1 # value for k1
    K[1] = k2

    for t in range(p.T - 2): # loop up to T-2 where k_t2 = k_T

        # rearanged Euler equation to solve k_t2
        k_t2 = (
            f(Z[t + 1], K[t + 1], p.alpha) + (1 - p.delta) * K[t + 1]
            - p.beta * (f(Z[t], K[t], p.alpha) + (1 - p.delta) * K[t] - K[t + 1]) # consumption in t
            * (deriv_f(Z[t + 1], K[t + 1], p.alpha) + (1 - p.delta)) # square brackets term
        )

        resources_t1 = f(Z[t + 1], K[t + 1], p.alpha) + (1 - p.delta) * K[t + 1] # max value for k_t1 where consumption = 0

        epsilon = 1e-8 # allow a tolerance to stop numerical instability
        # truncate k if outside of bounds
        k_t2 = max(k_t2, epsilon)
        k_t2 = min(k_t2, resources_t1 - epsilon)

        K[t + 2] = k_t2 # store k value in array and continue loop

    return K # return vector of all k value

def shoot(p: Params):

    z_star = 29 / 19 # z_star defined from geometric series
    k_star = k_steady_state(z_star, p)

    Z = create_z_vector(p.T) # create vector of Z values once and pass into function as needed

    k2_max = f(Z[0], p.k1, p.alpha) + (1 - p.delta) * p.k1 # max k2 when consumption = 0

    result = root_scalar(
        lambda k2: k_T(k2, Z, p)[-1] - k_star, # k_T is final value from vector
        bracket=[0.1, k2_max - 1e-4], # optimise within a tolerance
        method="brentq"
    )

    print(f"k2 = {result.root}")
    print(f"kstar = {k_star}")
    print(f"Error: {abs(k_T(result.root, Z, p)[-1] - k_star)}")

    return result.root


def plot_paths(k2_values, Z: np.ndarray, p: Params, k_star=None, k_target=None):
    """
    Plot capital paths with Seaborn styling and highlighting for the converging path.
    """
    # 1. Set the aesthetic theme
    sns.set_theme(style="whitegrid", palette="muted")
    plt.figure(figsize=(10, 6))
    
    periods = np.arange(1, p.T + 1)
    
    # 2. Create a color palette
    # Use a sequential palette or a single color with varying alphas
    colors = sns.color_palette("coolwarm", n_colors=len(k2_values))

    for i, k2 in enumerate(k2_values):
        K = k_T(k2, Z, p)
        
        # 3. Logic to highlight the "correct" or "converging" path
        # If k_target is provided, highlight the path that ends closest to it
        is_highlight = (k_target is not None and np.isclose(k2, k_target, atol=1e-5))
        
        if is_highlight:
            line_color = "forestgreen"
            alpha = 1.0
            linewidth = 3
            zorder = 5 # Bring to front
            label = fr"Converging $k_2 = {k2:.4f}$"
            linestyle = "-"
        else:
            line_color = colors[i]
            alpha = 1
            linewidth = 1.8
            zorder = 2
            label = fr"Non-converging $k_2 = {k2:.4f}$" # Hide redundant labels for background paths
            linestyle = "--"

        plt.plot(periods, K, color=line_color, alpha=alpha, 
                 linewidth=linewidth, zorder=zorder, label=label, linestyle = linestyle)

    # 4. Enhance the Steady State line
    if k_star is not None:
        plt.axhline(k_star, color="black", linestyle=":", 
                    linewidth=1.5, alpha=0.9, label=fr"Steady State $k^*$= {k_star:.4f}")

    # 5. Professional Touches
    plt.title("Dynamics of Capital Accumulation: Convergence Paths", fontsize=14, pad=15)
    plt.xlabel("Period ($t$)", fontsize=12)
    plt.ylabel("Capital Stock ($k_t$)", fontsize=12)
    
    # Remove the top and right spines for a "clean" look
    sns.despine()
    
    # Place legend outside or in a clean spot
    plt.legend(frameon=True, facecolor='white', loc='best')
    
    plt.tight_layout()
    
    plt.savefig("outputs/optimal_capital_path.pdf")
    plt.show()

def plot_shooting_function(k2_range, Z, p, k_star, k2_solved=None):
    """
    Plots Φ(k2) = k_T - k_star and highlights the solved k2 root.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # 1. Calculate the function values across the range
    k_Ts = [k_T(k2, Z, p)[-1] - k_star for k2 in k2_range]
    
    # 2. Plot the main shooting function curve
    # Using a gradient-compatible palette for the line
    plt.plot(k2_range, k_Ts, color="#2B9FEC", linewidth=2.0, 
             label=r"$k_T(k_2) - k^*$", zorder=3)

    # 3. Highlight the Zero-Error line
    plt.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.8)

    # 4. Highlight the Solved Root
    if k2_solved is not None:
        # Calculate the actual error at the solved point (should be ~0)
        solved_error = k_T(k2_solved, Z, p)[-1] - k_star
        
        # Vertical line at the solved k2
        plt.axvline(k2_solved, color="#F5591B", linestyle="--", linewidth=1.2, 
                    label=f"Solved $k_2 = {k2_solved:.5f}$", zorder=4)
        
        # Add a point marker at the intersection
        plt.scatter(k2_solved, solved_error, color="#F5591B", s=80, 
                    edgecolor='black', zorder=5, marker = "x")
        
        # Optional: Add a text annotation for clarity
        plt.annotate(f'Root found at\n$k_2$ = {k2_solved:.4f}', 
                     xy=(k2_solved, solved_error), 
                     xytext=(k2_solved + (max(k2_range)*0.05), solved_error + (max(k_Ts)*0.1)),
                     arrowprops=dict(arrowstyle='->', color='#F5591B'),
                     fontsize=16, color='black')

    # 5. Styling
    plt.title("Shooting Method: Root Finding for $k_2$", fontsize=15, pad=20)
    plt.xlabel("Initial Choice for $k_2$", fontsize=12)
    plt.ylabel("Terminal Deviation from Steady State", fontsize=12)
    
    # If the divergence is massive, symlog makes the root area more visible
    if max(np.abs(k_Ts)) > 100:
        plt.yscale('symlog', linthresh=1.0)
    
    sns.despine()
    plt.legend(frameon=True, loc='best')
    plt.tight_layout()
    plt.savefig("outputs/shooting_function.pdf")
    plt.show()

def plot_k_dynamics(K_solved, k_star):
    """
    K_solved: The capital vector after root-finding (np.array)
    k_star:   The theoretical long-run steady state
    """
    sns.set_theme(style="ticks", palette="viridis")
    plt.figure(figsize=(8, 8))
    
    # 1. 45-degree line (Reference for kt+1 = kt)
    limit = max(K_solved.max(), k_star) * 1.05
    plt.plot([1, limit], [1, limit], color="black", linestyle="--", alpha=0.3, label="45° line")
    
    # 2. Extract pairs (kt, kt+1)
    kt = K_solved[:-1]
    kt_next = K_solved[1:]
    
    # 3. Plot the path with a gradient to show time progression
    # The scatter dots will change color as time passes
    time_idx = np.arange(len(kt))
    scatter = plt.scatter(kt, kt_next, c=time_idx, cmap="magma", 
                          s=40, edgecolor='white', zorder=5, label="Transition Path")
    
    # Connect the dots with a subtle line to show the direction
    plt.plot(kt, kt_next, color="teal", alpha=0.4, linewidth=1.5, zorder=4)

    # 4. Highlight the Steady State Target
    plt.scatter(k_star, k_star, color="crimson", marker='x', s=250, 
                linewidth=0.8, zorder=6, label=f"Long-run $k^*$")

    # 5. Visual refinement
    plt.title("Transition Path of $(k_t, k_{t+1})$ to the steady state", fontsize=15, pad=20)
    plt.xlabel("Capital at time t ($k_t$)", fontsize=12)
    plt.ylabel("Capital at time t+1 ($k_{t+1}$)", fontsize=12)
    
    # Add colorbar to explain the dot colors
    cbar = plt.colorbar(scatter, shrink=0.8)
    cbar.set_label('Time Step', rotation=270, labelpad=15)
    
    # Force square aspect so the 45-degree line is actually 45 degrees
    plt.gca().set_aspect('equal')
    
    sns.despine(offset=10)
    plt.legend(frameon=True, loc='upper left')
    plt.tight_layout()
    plt.savefig("outputs/k_dynamics.pdf")
    plt.show()

def main():
    p = Params()
    k2 = shoot(p)

    Z = create_z_vector(p.T)
    z_star = 29 / 19
    k_star = k_steady_state(z_star, p)

    plot = True

    if plot == True:
        plot_paths([k2 * 0.995, k2 * 0.998, k2, k2 * 1.002, k2 * 1.005], Z, p, k_star, k2)

        k2_max = f(Z[0], p.k1, p.alpha) + (1 - p.delta) * p.k1
        k2_space = np.linspace(0.1, k2_max - 1e-4, 10000)
        plot_shooting_function(k2_space, Z, p, k_star, k2)

        K = k_T(k2, Z, p)
        plot_k_dynamics(K, k_star)

if __name__ == "__main__":
    main()