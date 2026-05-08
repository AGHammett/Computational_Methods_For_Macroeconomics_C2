import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from scipy.optimize import differential_evolution
from pathlib import Path

# -----------------------------------------------IMPORTING AND CLEANING THE DATA-----------------------------------------------

def load_and_clean_data(file_path):
    """
    Loads Prescott data, fills missing values, and renames columns for consistency.
    """
    df = pd.read_csv(file_path)

    #Fixes all missing values in the data set
    df["period"] = df["period"].ffill()
    df["country"] = df["country"].ffill()

    #Renaming columns so that they match my columns
    df = df.rename(columns={
        "h": "h_obs",
        "tau": "t",
        "c2y": "c_y"})

    #Renaming columns so that they have full country names rather than abbreviations
    country_map = {
        "DEU": "Germany",
        "FRA": "France",
        "ITA": "Italy",
        "CAN": "Canada",
        "GBR": "United Kingdom",
        "JPN": "Japan",
        "USA": "United States"}

    # dding the full names of countries to the clen data set
    df["country"] = df["country"].replace(country_map)
    
    return df

# -----------------------------------------------------------------CALIBRATING ALPHA & SIGMA--------------------------------------------------------------------


#Prescotts FOC from Q1
def prescott_foc_equilibrium_condition(tau, c_y, theta = 0.32, alpha = 1.7105):
     return 100 * (1 - theta) / ((alpha * c_y / (1 - tau)) + (1 - theta))

#FOC that characterises the optimal hours worked which is non-linear
def foc_equilibrium_condition(h, tau, c_y, alpha, sigma, theta = 0.32):
    return ((1 - tau) * (1 - theta)) / (c_y * h) - alpha * (100 - h) ** -sigma


#Using these values to find a sign change for h in order to use brentq as a root finding method
def check_sign_change():
    tau = 0.44
    c_y = 0.83
    alpha = 1.7105
    sigma = 1.0
    theta = 0.32

    #Checking for a sign change to use brentq
    print("\nFOC at h=1:", foc_equilibrium_condition(1, tau, c_y, alpha, sigma, theta))
    print("FOC at h=21:", foc_equilibrium_condition(21, tau, c_y, alpha, sigma, theta))
    print("FOC at h=99:", foc_equilibrium_condition(99, tau, c_y, alpha, sigma, theta))

#Creating a root finding function to solve for h
def solve_h_brentq(tau, c_y, alpha, sigma, theta = 0.32):
    return brentq(foc_equilibrium_condition, 1e-6, 99.99, args = (tau, c_y, alpha, sigma, theta))


#Creating the loss function for alpha and sigma to be calibrated on
def loss_function(parameters, df, theta = 0.32): #Theta is kept fixed
    alpha, sigma = parameters

    #Placing a large penalty for invalid values (negatives) 
    #This should stop certain optimisers from deviating past bounds
    if alpha <= 0 or sigma <= 0:
        return 1e10
    
    #New predict column is added to the data frame
    h_pred = df.apply( #Lambda is used here to create a temporary solve h function
        lambda row: solve_h_brentq(row["t"], row["c_y"], alpha, sigma, theta),
        axis = 1)
    #Below is the formula to calculate the losses between predicted and actual values
    loss = ((df["h_obs"] - h_pred) ** 2).sum()
    return float(loss) #Expecting the loss to a number with a decimal point which improves precision


#A variety of different optimisers

def grid_search_a_s(df, density):
    alpha_grid = np.linspace(500, 20000, density) #Creating a large grid to search up for alpha and sigma
    sigma_grid = np.linspace(1.5, 4.0, density)
    best_grid_loss = np.inf #infinite grid 
    best_grid_alpha = None
    best_grid_sigma = None

    for alpha in alpha_grid: # loop over grid
        for sigma in sigma_grid:
            try:
                loss = loss_function([alpha, sigma], df)
            except ValueError: #This stops the function from breaking if we encounter an error
                loss = np.inf #If we get an error, loss becomes infinity

            if loss < best_grid_loss: #Constantly the previous loss to the next loss, finding the smallest one
                best_grid_loss = loss #The smallest one will then become the best loss (smallest loss)
                best_grid_alpha = alpha #The associated alpha and sigma from that loss are the calibrated parameters 
                best_grid_sigma = sigma
    
    res_dict = { # result dict has all info needed - this pattern will be used in all methods
        "Method": "Grid Search",
        "Alpha": best_grid_alpha,
        "Sigma": best_grid_sigma,
        "Loss": best_grid_loss,
        "Success": True
    }
    
    return res_dict

def minimise_loss_group(df, methods: list[dict], guess, hybrid_method: str = None):

    results = []
    for spec in methods:
        method = spec["method"]
        method_bounds = spec["bounds"]

        res = minimize(
            loss_function,
            x0 = guess,
            args = (df,),          # important: args should usually be a tuple
            method = method,
            bounds = method_bounds
        )

        res_dict = {
            "Method": f"{hybrid_method} {method}".strip(),
            "Alpha": res.x[0],
            "Sigma": res.x[1],
            "Loss": res.fun,
            "Success": res.success
        }

        results.append(res_dict)

    return results

def minimise_diff_ev(df, bounds, seed = 123):

    res_de = differential_evolution(
        lambda params: loss_function(params, df),
        bounds= bounds,
        tol = 1e-8, #The tolerence for the optimisation to end 
        seed = seed) # random seed controls rng 

    res_dict = {
            "Method": "Differential Evolution",
            "Alpha": res_de.x[0],
            "Sigma": res_de.x[1],
            "Loss": res_de.fun,
            "Success": res_de.success
        }

    return res_dict

def run_optimisation(df):

    # set up parameters
    grid_density = 80
    start_guess = (10000, 2)
    bounds = [(0.01, 100000), (0.01, 10)]
    n_random_starts = 50 # no of random nelder mead iterations
    np.random.seed(123) # seed controls randomiser for nelder mead points

    results = [] # create empty list for results

    methods = [ # list of methods that can be used in minimise_loss_group
        {"method": "Nelder-Mead", "bounds": None},
        {"method": "BFGS", "bounds": None},
        {"method": "L-BFGS-B", "bounds": bounds},
        ]

    grid_res = grid_search_a_s(df, grid_density)
    results.append(grid_res) # add to results
    grid_alpha = grid_res["Alpha"] # store points for use as starting points later
    grid_sigma = grid_res["Sigma"]

    # get results for using simply methods
    single_results = minimise_loss_group(df, methods, start_guess, "")
    results += single_results# += combines the 2 lists

    # get results when using grid search min as a starting point
    grid_start_results = minimise_loss_group(df, methods, (grid_alpha, grid_sigma), "Grid Search")
    results += grid_start_results 

    # show exmaple of how wrong starting guess get's wrong point
    wrong_start_results = minimise_loss_group(df, [methods[0]], (10, 2), "Bad Start")
    results += wrong_start_results

    # get results with diff ev routine
    diff_ev_results = minimise_diff_ev(df, bounds)
    results.append(diff_ev_results)
    diff_alpha = diff_ev_results["Alpha"] # store points for starting points
    diff_sigma = diff_ev_results["Sigma"]

    # run bounded quasi newton on diff ev starting point
    diff_ev_qn_results = minimise_loss_group(df, [methods[2]], (diff_alpha, diff_sigma), "Diff Ev")
    results += diff_ev_qn_results

    # randomise nelder mead starts for robustness
    best_rand_guess = 1e10 # high value will be replaced on first iteration
    best_rand_res = None 
    for i in range(n_random_starts):

        # randomise start points
        alpha_start = np.random.uniform(100, 100000) 
        sigma_start = np.random.uniform(0.5, 10)

        try: # handle errors when foc fails to have a sign change
            rand_nm_res = minimise_loss_group(df, [methods[0]], (alpha_start, sigma_start), "Random Start")[0] # need to recover results from list
        except ValueError: # brent q throws value error so it handles that
            print(f"Thrown Error Iteration no {i} at Alpha {alpha_start} & Sigma {sigma_start}")
            continue # skip iteration after error raised

        if rand_nm_res["Loss"] < best_rand_guess:
            best_rand_guess = rand_nm_res["Loss"]
            best_rand_res = rand_nm_res
    
    results.append(best_rand_res)
    print(f"Diff Ev best Loss: {diff_ev_results['Loss']}") # use single quotes for nested strings
    print(f"Random NM best Loss: {best_rand_res['Loss']}")

    df_results = pd.DataFrame(results)

    #Rounding all results for presentation
    df_results[["Alpha", "Sigma", "Loss"]] = df_results[["Alpha", "Sigma", "Loss"]].round(4)
    latex_table = df_results.to_latex(index=False, float_format="%.4f")
    print(latex_table)
    print(df_results)
    return df_results

#-------------------------------------------REPRODUCING TABLE WITH ALPHA & SIGMA CALIBRATED-----------------------------------------

def create_predictions(df, alpha, sigma):
#Creating a new data frame for the jointly calibrated variables to be presented in a table
    df_jointly_calibrated_table = df.copy()

    #Renaming columns to match lecture slides
    df_jointly_calibrated_table = df_jointly_calibrated_table.rename(columns = {
        "period": "Period",
        "country": "Country",
        "h_obs": "Actual",
        "t": "tau",
        "c_y": "c_y"})

    #Changing the period labels to match the lecture slides
    df_jointly_calibrated_table["Period"] = df_jointly_calibrated_table["Period"].replace({
        "1993-1996": "1993-96",
        "1970-1974": "1970-74"})

    #Adding the top row of labels for the table
    df_jointly_calibrated_table = df_jointly_calibrated_table[["Period", "Country", "Actual", "tau", "c_y"]]

    #Adding the predicted labour supply using differential evolution and bounded quasi hybrid to calibrate alpha and sigma
    df_jointly_calibrated_table["Predict"] = df_jointly_calibrated_table.apply(
        lambda row: solve_h_brentq(row["tau"], row["c_y"], alpha, sigma),
        axis=1).round(1)

    #Adding in  a difference column
    df_jointly_calibrated_table["Difference"] = (df_jointly_calibrated_table["Predict"] - df_jointly_calibrated_table["Actual"]).round(1)

    print(df_jointly_calibrated_table)

    #Converting pandas table to latex so it can be easily used in document 
    latex_table_1a = df_jointly_calibrated_table.to_latex(index = False, float_format = "%.1f") #Rounds to 1dp
    print(latex_table_1a)

    return df_jointly_calibrated_table

#-------------------------------------------------------PLOTTING PREDICTED VS CALIBRATED VALUES---------------------------------------------

def plot_model_fit(df, period, x_lim, y_lim, save_name = None, title=None):
    """
    Plots Predicted vs Actual hours for a specific period.
    """
    fig, ax = plt.subplots()

    labels_map = {
        "Germany": "DEU",
        "France": "FRA",
        "Italy": "ITA",
        "Canada": "CAN",
        "United Kingdom": "GBR",
        "Japan": "JPN",
        "United States": "USA"}

    df_period = df[df["Period"] == period].copy()
    df_period["label"] = df_period["Country"].map(labels_map)

    print(f"\nPlotting for {period}:\n", df_period)
    ax.scatter(df_period["Actual"], df_period["Predict"], color="red", s=30)

    # Adding in the country abbreviations to the scatter plot
    for _, row in df_period.iterrows():
        ax.text(
            row["Actual"] + 0.1,   # Shifting actual point slightly right to avoid a messy graph
            row["Predict"] + 0.1,  # Shift predicted point up slightly
            row["label"],          # Adding the label to plot
            fontsize=10)

    # Adding in the 45 degree dashed line
    xmin = min(df_period["Actual"].min(), df_period["Predict"].min()) - 1
    xmax = max(df_period["Actual"].max(), df_period["Predict"].max()) + 1
    ax.plot([xmin, xmax], [xmin, xmax], linestyle="--", color="black")

    # Limiting both axes
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    ax.set_title(title if title else f"Model Fit: {period}")
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")

    fig.tight_layout()

    if save_name:
        fig.savefig(f"{save_name}.pdf", dpi=300, bbox_inches="tight")
    plt.show()

#----------------------------------------------CREATING AN IMPROVEMENT COLUMN FOR PRESCOTT AND JOINT CALIBRATED MODEL----------------------------------------------------


def create_improvement_df(df_crra, df_prescott):
#This code is coppied from Q1 script as it uses the exact same data with the exception of changing the Pandas DataFrame name 

    df_prescott = df_prescott.copy()
    #Renaming columns to match lecture slides
    df_prescott = df_prescott.rename(columns = {
        "period": "Period",
        "country": "Country",
        "h_obs": "Actual",
        "t": "tau",
        "c_y": "c_y"})

    #Making time period labels match lecture slides
    df_prescott["Period"] = df_prescott["Period"].replace({
        "1993-1996": "1993-96",
        "1970-1974": "1970-74"})

    #Keeping only the necessary columns required
    df_prescott = df_prescott[["Period", "Country", "Actual", "tau", "c_y"]]

    #Adding in predicted labour supply using Q1 calibrated alpha
    df_prescott["Predict"] = df_prescott.apply(
        lambda row: prescott_foc_equilibrium_condition(row["tau"], row["c_y"]), axis=1).round(1)

    #Adding in the difference column
    df_prescott["Difference"] = (df_prescott["Predict"] - df_prescott["Actual"]).round(1)

    #Creating a new data frame to compare the data from Prescotts model and the CRRA model where parameters were jointly calibrated
    df_compare = df_crra.copy()

    df_compare["Prescott Difference"] = df_prescott["Difference"]
    df_compare["CRRA Difference"] = df_crra["Difference"]

    #Calculating the difference between the Prescott data and the CRRA data
    df_compare["Improvement"] = df_compare["Prescott Difference"].abs() - df_compare["CRRA Difference"].abs()

    #Creating the columns for the new table comparing the values
    df_compare = df_compare[["Period", "Country", "Prescott Difference", "CRRA Difference", "Improvement"]]

    print(df_compare.to_latex(index=False, float_format="%.1f")) #Latex form 
    print("\n",df_compare)

    return df_compare


#---------------------------------------------GRAPHING THE SURFACE AND CONTOUR OF THE LOSS FUNCTION---------------------------------------

def create_point_styles(results):
# Define style for each method for points on 3D graph
    style_map = { # Dictionary with unique style for each method
        "Grid Search": {"name": "Grid Search", "colour": "green", "marker": "v"},
        "Nelder-Mead": {"name": "Nelder-Mead", "colour": "pink", "marker": "o"},
        "BFGS": {"name": "BFGS", "colour": "black", "marker": "X"},
        "L-BFGS-B": {"name": "L-BFGS-B", "colour": "blue", "marker": "^"},
        "Grid Search Nelder-Mead": {"name": "Grid Search + Nelder-Mead", "colour": "purple", "marker": "P"},
        "Grid Search BFGS": {"name": "Grid Search + BFGS", "colour": "orange", "marker": "s"},
        "Grid Search L-BFGS-B": {"name": "Grid Search + L-BFGS-B", "colour": "cyan", "marker": "D"},
        "Bad Start Nelder-Mead": {"name": "Bad Start + Nelder-Mead", "colour": "red", "marker": "x"},
        "Differential Evolution": {"name": "Differential Evolution", "colour": "gray", "marker": "*"},
        "Diff Ev L-BFGS-B": {"name": "Differential Evolution + L-BFGS-B", "colour": "brown", "marker": "h"},
        "Random Start Nelder-Mead": {"name": "Random Start + Nelder-Mead", "colour": "magenta", "marker": "d"}
    }

    # Create dict using results and style map
    optimiser_points = {
        style_map[res["Method"]]["name"]: {
            "alpha": res["Alpha"],
            "sigma": res["Sigma"],
            "colour": style_map[res["Method"]]["colour"],
            "marker": style_map[res["Method"]]["marker"],
            "size": 120
        }
        for res in results if res["Method"] in style_map
    }

    return optimiser_points

#Alphas and sigmas are looped into the loss function and the loss value is stored which computes the plotted surface 
def plot_loss_surface(df, loss_function, alpha_range, sigma_range, optimiser_points = None, save_name = None):
    
    alpha_vals = np.linspace(*alpha_range) #Turning the 1D array into a 2D matracies
    sigma_vals = np.linspace(*sigma_range)

    A, S = np.meshgrid(alpha_vals, sigma_vals)
    Z = np.zeros_like(A) #This is creating an empty container for the loss values 

    for i in range(A.shape[0]): #Creating a loop with alpha and sigma running into the loss function 
        for j in range(A.shape[1]):
            try:
                Z[i, j] = loss_function([A[i, j], S[i, j]], df) #Taking a specific alpha and sigma and computing the loss
            except ValueError:
                Z[i, j] = np.nan #Stops the function from breaking and records any alphas and sigmas that dont have a sign change to write (nan)
                        #Matplot will ignore any (nan) so it does not affect the plotting 

    fig = plt.figure(figsize = (10, 7)) #This is creating an empty canvas with the associated dimensions
    ax = fig.add_subplot(111, projection="3d") #This transforms the axis to 3D

    ax.plot_surface(A, S, Z, alpha = 0.7) #alpha = 0.7 so that they are slightly raised through the surface, allowing for better visualisation
    ax.view_init(elev = 17, azim = 192) #Azim controls the horizontal rotation whilst elev controls the vertical elevation (Birds eye view)

    #Only the points run through the function will be plotted
    if optimiser_points is not None: #Intentially seperated from the main function itself allowing for easier debugging if required
        for name, point in optimiser_points.items(): #All of the features below can be changed individually in the section below
            alpha_hat = point["alpha"]
            sigma_hat = point["sigma"]
            colour = point["colour"]
            marker = point["marker"]
            size = point["size"]

            loss_hat = loss_function([alpha_hat, sigma_hat], df) #This allows the points to be on the loss surface

            ax.scatter( #All of the features below can be changed individually in the section below
                alpha_hat,
                sigma_hat,
                loss_hat,
                color = colour,
                marker = marker,
                edgecolor = "black",
                s = size,
                label = name)

    ax.set_xlabel("Alpha", labelpad = 15) #Pushes the lables slightly away from the axes so they do no merge with the units
    ax.set_ylabel("Sigma", labelpad = 15)
    ax.set_zlabel("Loss", labelpad = 15)
    ax.set_title("Loss Surface")

    ax.legend(fontsize = 8, loc = "upper left", bbox_to_anchor = (1.05, 1)) #Locking the legend in place and resizing it

    fig.tight_layout()
    if save_name: # optionally save
        fig.savefig(save_name, dpi = 300, bbox_inches = "tight", pad_inches = 0.3)
    plt.show()

    return A, S, Z

def plot_loss_heatmap(A, S, Z, optimiser_points, save_name = False):
    fig, ax = plt.subplots(figsize = (8, 6)) #A new canvas is created for the contour plot with the associated dimensions

    #We are normalising the loss curve so that the lowest value essentially becomes 0 on the heat map
    Z_min = np.nanmin(Z)
    Z_shifted = Z - Z_min

    levels = np.linspace(0, np.percentile(Z_shifted, 10), 100) #Only focuses on the 10% of the loss surface
                                                            #This allows for greater precision when looking at the contour lines

    #Below are parameters to smothen the colour shading and reduce the number of black contour lines making results better presented
    levels_fill = np.linspace(0, np.nanpercentile(Z_shifted, 10), 100) #np.nanpercentile ignores any nan's that have been pulled through
    levels_lines = np.linspace(0, np.nanpercentile(Z_shifted, 10), 12)

    contour = ax.contourf(A, S, Z_shifted, levels = levels_fill) #Creates a smooth colour heat map
    ax.contour(A, S, Z_shifted, levels = levels_lines, colors = "black", linewidths = 0.5) #draws on the contour lines with the associated features
    fig.colorbar(contour, ax = ax, label = "Loss Above Minimum") #Labeling the colour bar 

    #Adding in the optimiser points using same dictionary as the surface plot
    for name, point in optimiser_points.items(): #Each sigma and alpha will provide with a loss value, which is then plotted

        ax.scatter( #Same parameters are changed above for both plots
            point["alpha"],
            point["sigma"],
            color = point["colour"],
            edgecolor = "black",
            s = point["size"],
            marker = point["marker"],
            label = name)

    ax.set_xlim(9000, 20000) #Limiting the axis for better clarity of results
    ax.set_ylim(2.95, 3.1)

    ax.set_xlabel("Alpha")
    ax.set_ylabel("Sigma")

    ax.set_title("Loss Function Contour")
    ax.legend(loc = "lower right") # corner has least info relevant to plot

    fig.tight_layout()

    if save_name: # optional save
        fig.savefig(save_name, dpi = 300, bbox_inches = "tight", pad_inches = 0.5)

    plt.show()


def main():

    # set data path, import and clean data
    DATA_PATH = Path("q2data/data_prescott.csv")
    df_clean = load_and_clean_data(DATA_PATH)

    #Creating a new data frame for calibrating using CRRA utility funciton
    df_calib_crra = df_clean.copy()

    # run all optimisation methods - results are a list of dicts
    df_optim_results = run_optimisation(df_calib_crra)

    # find best loss row for prediction
    best_loss_row_idx = df_optim_results["Loss"].idxmin()
    best_alpha = df_optim_results.loc[best_loss_row_idx, "Alpha"]
    best_sigma = df_optim_results.loc[best_loss_row_idx, "Sigma"]
    print(f"Lowest Loss value found by {df_optim_results.loc[df_optim_results['Loss'].idxmin()]}")

    df_jointly_calibrated_table = create_predictions(df_calib_crra, best_alpha, best_sigma)

    # create improvememnt df
    create_improvement_df(df_jointly_calibrated_table, df_clean)

    plot_graphs = True # set to False to skip graphing
    if plot_graphs == True:
        # Plotting data for 1993 - 1996
        plot_model_fit(df_jointly_calibrated_table, "1993-96", (16, 28), (16, 28), save_name = "Model Fit 1993-96", title="CRRA Model Fit: 1993-96")

        # Plotting data for 1970 - 1974
        plot_model_fit(df_jointly_calibrated_table, "1970-74", (18, 30), (18, 34), save_name = "Model Fit 1970-74", title="CRRA Model Fit: 1970-74")
    
        # turn results into a dict for use in plot styles
        results = df_optim_results.to_dict(orient="records")
        #create style dictionary
        optimiser_points = create_point_styles(results)
            
        #Plotting the ranges used for both surface and contour
        alpha_plot_range = (2000, 20000, 150)
        sigma_plot_range = (2.0, 4.0, 150)
            
        #Below is the line which calls the function and plots the loss surface
        A, S, Z = plot_loss_surface(df_calib_crra, loss_function, alpha_plot_range, sigma_plot_range, optimiser_points, save_name = "Loss surface.pdf")

        plot_loss_heatmap(A, S, Z, optimiser_points, save_name = "Contour of loss surface.pdf")

if __name__ == "__main__":
    main()