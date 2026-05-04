import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from q4 import Params, k_T, f, shoot, create_z_vector, k_steady_state

def plot_paths(k2_values, Z: np.ndarray, p: Params, k_star=None, k_target=None):
    """
    Plot capital paths with Seaborn styling and highlighting for the converging path.
    """
   # set seaborn theme
    sns.set_theme(style="whitegrid", palette="muted")
    plt.figure(figsize=(10, 6))
    # create an array for time periods starting at 1
    periods = np.arange(1, p.T + 1)
    
    # fading colour pattern
    colors = sns.color_palette("coolwarm", n_colors=len(k2_values))
    # loop through different starting k values
    for i, k2 in enumerate(k2_values):
        K = k_T(k2, Z, p)

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
            line_color = colors[i] # take colour from gradient palette
            alpha = 1
            linewidth = 1.8
            zorder = 2
            label = fr"Non-converging $k_2 = {k2:.4f}$" # Hide redundant labels for background paths
            linestyle = "--"

        plt.plot(periods, K, color=line_color, alpha=alpha, 
                 linewidth=linewidth, zorder=zorder, label=label, linestyle = linestyle)

    # # plot steady state vale
    if k_star is not None:
        plt.axhline(k_star, color="#4B0082", linestyle="-.", 
                    linewidth=1.8, alpha=1.0, label=fr"Steady State $k^*$= {k_star:.4f}")


    plt.title("Dynamics of Capital Accumulation: Convergence Paths", fontsize=14, pad=15)
    plt.xlabel("Period ($t$)", fontsize=12)
    plt.ylabel("Capital Stock ($k_t$)", fontsize=12)
    sns.despine()
    plt.legend(frameon=True, facecolor='white', loc='best')
    
    plt.tight_layout()
    
    if __name__ == "__main__": # don't save when run from q4.py
        plt.savefig("outputs/optimal_capital_path.pdf")

    plt.show()

def plot_shooting_function(k2_range, Z, p, k_star, k2_solved=None):
    """
    Plots p(k2) = k_T - k_star and highlights the solved k2 root.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))
    
    # calculate the function values across the range
    k_Ts = [k_T(k2, Z, p)[-1] - k_star for k2 in k2_range]
    
    # plot shooting function curve
    plt.plot(k2_range, k_Ts, color="#2B9FEC", linewidth=2.0, 
             label=r"$k_T(k_2) - k^*$", zorder=3)

    # plot the zero error line
    plt.axhline(0, color="black", linestyle="-", linewidth=0.8, alpha=0.8)

    # highlight the solved root
    if k2_solved is not None:
        # Calculate the actual error at the solved point (should be ~0)
        solved_error = k_T(k2_solved, Z, p)[-1] - k_star
        
        plt.axvline(k2_solved, color="#F5591B", linestyle="--", linewidth=1.2, 
                    label=f"Solved $k_2 = {k2_solved:.5f}$", zorder=4)
        
        # point at intersction
        plt.scatter(k2_solved, solved_error, color="#F5591B", s=80, 
                    edgecolor='black', zorder=5, marker = "x")
        
        plt.annotate(f'Root found at\n$k_2$ = {k2_solved:.4f}', 
                     xy=(k2_solved, solved_error), 
                     xytext=(k2_solved + (max(k2_range)*0.05), solved_error + (max(k_Ts)*0.1)),
                     arrowprops=dict(arrowstyle='->', color='#F5591B'),
                     fontsize=16, color='black')

    plt.title("Shooting Method: Root Finding for $k_2$", fontsize=15, pad=20)
    plt.xlabel("Initial Choice for $k_2$", fontsize=12)
    plt.ylabel("Terminal Deviation from Steady State", fontsize=12)
    
    if max(np.abs(k_Ts)) > 100:
        plt.yscale('symlog', linthresh=1.0)
    
    sns.despine()
    plt.legend(frameon=True, loc='best')
    plt.tight_layout()
    plt.savefig("outputs/shooting_function.pdf")
    plt.show()

def plot_k_dynamics(K_solved, k_star):
    """
    Plots k in adjacent periods
    """
    sns.set_theme(style="ticks", palette="viridis")
    plt.figure(figsize=(8, 8))
    
    # 45 degree line
    limit = max(K_solved.max(), k_star) * 1.05
    plt.plot([1, limit], [1, limit], color="black", linestyle="--", alpha=0.3, label="45° line")
    
    # get pairs from list
    kt = K_solved[:-1]
    kt_next = K_solved[1:]
    
    # plot using a gradient to colour points
    time_idx = np.arange(len(kt))
    scatter = plt.scatter(kt, kt_next, c=time_idx, cmap="magma", 
                          s=40, edgecolor='white', zorder=5, label="Transition Path")
    
    # connect points
    plt.plot(kt, kt_next, color="teal", alpha=0.4, linewidth=1.5, zorder=4)

    # highlight steady state
    plt.scatter(k_star, k_star, color="crimson", marker='x', s=250, 
                linewidth=0.8, zorder=6, label=f"Long-run $k^*$")

    plt.title("Transition Path of $(k_t, k_{t+1})$ to the steady state", fontsize=15, pad=20)
    plt.xlabel("Capital at time t ($k_t$)", fontsize=12)
    plt.ylabel("Capital at time t+1 ($k_{t+1}$)", fontsize=12)
    
    cbar = plt.colorbar(scatter, shrink=0.8)
    cbar.set_label('Time Step', rotation=270, labelpad=15)
    
    plt.gca().set_aspect('equal')
    
    sns.despine(offset=10)
    plt.legend(frameon=True, loc='upper left')
    plt.tight_layout()
    plt.savefig("outputs/k_dynamics.pdf")
    plt.show()

def main():
    p = Params() # instantiate parameter object
    k2 = shoot(p) # get k2 for ss solution

    Z = create_z_vector(p.T) # create TFP vector
    z_star = 29 / 19
    k_star = k_steady_state(z_star, p)
    k2_max = f(Z[0], p.k1, p.alpha) + (1 - p.delta) * p.k1 # upped bound

    # Generate all plots
    plot_paths([k2 * 0.995, k2 * 0.998, k2, k2 * 1.002, k2 * 1.005], Z, p, k_star, k2)
    
    k2_space = np.linspace(0.1, k2_max - 1e-4, 10000)
    plot_shooting_function(k2_space, Z, p, k_star, k2)

    K = k_T(k2, Z, p)
    plot_k_dynamics(K, k_star)

if __name__ == "__main__":
    main()
