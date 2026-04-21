import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar
from q2 import h

def load_data(path):

    return df

def calibrate_alpha(df, alpha_guess = 0, theta = 1.77):

        alpha = alpha_guess
        df["h_mod"] = h(theta, alpha, df["t"], df["c_o"])

    return 

def loss_function(h_obs: pd.Series, h_mod: pd.Series):
    """
    Arguments - h_obs, h_mod -> pandas columns
    
    Outputs - loss -> float (scalar)
    """
    #elementwise loss
    loss_series = (h_obs - h_mod)**2

    return float(sum(loss_series))

