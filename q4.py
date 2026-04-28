import numpy as np
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

        max_k = f(Z[t + 1], K[t + 1], p.alpha) + (1 - p.delta) * K[t + 1] # max value for k_t1 where consumption = 0

        epsilon = 1e-8 # allow a tolerance to stop numerical instability
        # truncate k if outside of bounds
        k_t2 = max(k_t2, epsilon)
        k_t2 = min(k_t2, max_k - epsilon)

        K[t + 2] = k_t2 # store k value in array and continue loop

    return K # return vector of all k value

def shoot(p: Params):

    z_star = 29 / 19 # z_star defined from geometric series
    k_star = k_steady_state(z_star, p)

    Z = create_z_vector(p.T) # create vector of Z values once and pass into function as needed

    k2_max = f(Z[0], p.k1, p.alpha) + (1 - p.delta) * p.k1 # max k2 when consumption = 0

    result = root_scalar(
        lambda k2: k_T(k2, Z, p)[-1] - k_star, # k_T is final value from vector
        bracket=[0.1, k2_max - 1e-4], # optimise within brackets defined by k values
        method="brentq"
    )
    # print results
    print(f"k2 = {result.root}")
    print(f"kstar = {k_star}")
    print(f"Error: {abs(k_T(result.root, Z, p)[-1] - k_star)}")

    return result.root

def main():
    import q4_plots
    p = Params() # instantiate parameters object
    k2 = shoot(p) # get steady state k2

    # create variable required for plotting
    Z = create_z_vector(p.T)
    z_star = 29 / 19
    k_star = k_steady_state(z_star, p)

    plot = True
    if plot == True:
        q4_plots.plot_paths([k2 * 0.995, k2 * 0.998, k2, k2 * 1.002, k2 * 1.005], Z, p, k_star, k2)

if __name__ == "__main__":
    main()
