import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from dataclasses import dataclass

@dataclass(frozen=True)
class Params:
    alpha: float = 0.3
    beta: float = 0.96
    delta: float = 0.1
    k1: float = 1.0
    T: int = 30

def create_z_vector(T): 
    """
    Creates a vector of length T where T[i] is z_i
    """
    z_singles = (1 / 2.9) ** np.arange(T) # calculate the values for the ith term of the summation
    return np.cumsum(z_singles) # cumulatively sum each element for z_i

def k_steady_state(alpha, beta, delta, z_star):
    """
    Returns k* for given set of parameters
    """
    rho = (1 / beta) - 1
    return ((rho + delta) / (z_star * alpha)) ** (1 / (alpha - 1))

def f(z, k, alpha): return z * k ** alpha # production function

def deriv_f(z, k, alpha): return alpha * z * k ** (alpha - 1) # derivative of production function wrt k

def c(F, z, k_t, k_t1, alpha, delta): return F(z, k_t, alpha) + (1 - delta) * k_t - k_t1

def u(c): return np.log(c)

def delta_u(c): return  1 / c

def k_T(k1, k2, Z, T, alpha, beta, delta):
    """
    Uses Euler equation to solve for k_{t+2} recursively up to k_T.
    Returns vector of k values
    """

    # create k vector and fill in initial values
    K = np.empty(T)
    K[0]= k1
    K[1] = k2

    for t in range(T - 2): # loop up to T-2 where k_t2 = k_T

        # rearanged Euler equation to solve k_t2
        k_t2 = (
            f(Z[t + 1], K[t + 1], alpha) 
            + (1 - delta) * K[t + 1]
            - beta * (deriv_f(Z[t + 1], K[t + 1], alpha) + (1 - delta)) 
            * (f(Z[t], K[t], alpha) + (1 - delta) * K[t] - K[t + 1]) # consumption in t
        )

        resources_t1 = f(Z[t + 1], K[t + 1], alpha) + (1 - delta) * K[t + 1] # max value for k_t1 where consumption = 0

        epsilon = 1e-8 # allow a tolerance to stop numerical instability
        # truncate k if outside of bounds
        k_t2 = max(k_t2, epsilon)
        k_t2 = min(k_t2, resources_t1 - epsilon)

        K[t + 2] = k_t2 # store k value in array and continue loop

    return K # return vector of all k value

def shoot(k1, alpha, delta, beta, T):
    z_star = 29 / 19
    k_star = k_steady_state(alpha, beta, delta, z_star)

    
    Z = create_z_vector(T) # create vector of Z values once and pass into function as needed

    k2_max = f(Z[0], k1, alpha) + (1 - delta) * k1 # max k2 when consumption = 0

    result = root_scalar(
        lambda k2: k_T(k1, k2, Z,  T, alpha, beta, delta)[-1] - k_star, # k_T is final value from vector
        bracket=[0.1, k2_max - 1e-4], # optimise within a tolerance
        method="brentq"
    )

    print(f"k2 = {result.root}")
    print(f"kstar = {k_star}")

    return result.root

def plot_

def main():
    shoot(k1=1, alpha=0.3, delta=0.1, beta=0.96, T=30)

if __name__ == "__main__":
    main()