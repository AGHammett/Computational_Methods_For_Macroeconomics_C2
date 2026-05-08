import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.optimize import minimize_scalar
from scipy.optimize import minimize
from scipy.optimize import differential_evolution


# -----------------------------------------------IMPORTING AND CLEANING THE DATA-----------------------------------------------



#Importing Prescott data 
df_raw = pd.read_csv('/Users/mackmarin/Documents/Econ Year 3/EconPython/Computational Methods/Course Work 2/data_prescott.csv')

#Creating a new data frame for the cleaned data
df_clean= df_raw.copy()

#Fixes all missing values in the data set
df_clean["period"] = df_clean["period"].ffill()
df_clean["country"] = df_clean["country"].ffill()

#Renaming columns so that they match my columns
df_clean = df_clean.rename(columns = {
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

#Adding the full names of countries to the clen data set
df_clean["country"] = df_clean["country"].replace(country_map)




# -----------------------------------------------------------------CALIBRATING ALPHA & SIGMA--------------------------------------------------------------------





#Creating a new data frame for calibrating using CRRA utility funciton
df_calib_crra = df_clean.copy()


#Prescotts FOC from Q1
def prescott_foc_equilibrium_condition(tau, c_y, theta = 0.32, alpha = 1.7105):
     return 100 * (1 - theta) / ((alpha * c_y / (1 - tau)) + (1 - theta))



#FOC that characterises the optimal hours worked which is non-linear
def foc_equilibrium_condition(h, tau, c_y, alpha, sigma, theta = 0.32):
    return ((1 - tau) * (1 - theta)) / (c_y * h) - alpha * (100 - h) ** -sigma


#Using these values to find a sign change for h in order to use brentq as a root finding method
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


#A variety of different optimiser

# 1) L-BFGS-B - bounded quasi-Newton
res_lbfgsb = minimize(
    loss_function,
    x0=[1.7, 1.0], #This is the initial guess 
    args=(df_calib_crra,), #data set with the non-linear h term
    method="L-BFGS-B",
    bounds=[(0.01, 100000), (0.01, 10)])

alpha_lbfgsb, sigma_lbfgsb = res_lbfgsb.x #Extracting the alpha and sigma values that minimise the function fro the variable 


# 2) Nelder-Mead - single starting point
res_nm_single = minimize(
    loss_function,
    x0=[3000, 2.0], #Initial guess
    args=(df_calib_crra,),
    method="Nelder-Mead")

alpha_nm_single, sigma_nm_single = res_nm_single.x


# 3) BFGS - unbounded quasi-Newton
res_bfgs = minimize(
    loss_function,
    x0=[1.7, 1.0], #Initial guess
    args=(df_calib_crra,),
    method="BFGS")

alpha_bfgs, sigma_bfgs = res_bfgs.x


# 4) Grid search
alpha_grid = np.linspace(500, 20000, 80) #Creating a large grid to search up for alpha and sigma
sigma_grid = np.linspace(1.5, 4.0, 80)
best_grid_loss = np.inf #infinite grid 
best_grid_alpha = None
best_grid_sigma = None

for alpha in alpha_grid:
    for sigma in sigma_grid:
        try:
            loss = loss_function([alpha, sigma], df_calib_crra)
        except ValueError: #This stops the function from breaking if we encounter an error
            loss = np.inf #If we get an error, loss becomes infinity

        if loss < best_grid_loss: #Constantly the previous loss to the next loss, finding the smallest one
            best_grid_loss = loss #The smallest one will then become the best loss (smallest loss)
            best_grid_alpha = alpha #The associated alpha and sigma from that loss are the calibrated parameters 
            best_grid_sigma = sigma


# 5) Hybrid - grid search + BFGS
res_grid_bfgs = minimize(
    loss_function,
    x0=[best_grid_alpha, best_grid_sigma], #Uses the grid search results from above as the initial guess
    args=(df_calib_crra,),
    method="BFGS") #Quasi-Newton is applied to the initial gris search results 

alpha_grid_bfgs, sigma_grid_bfgs = res_grid_bfgs.x


# 6) Differential evolution
res_de = differential_evolution(
    lambda params: loss_function(params, df_calib_crra),
    bounds=[(100, 30000),(0.5, 6.0)],
    tol=1e-8, #The tolerence for the optimisation to end 
    seed=123) #Optimisation follows random paths but from the same starting point 

alpha_de, sigma_de = res_de.x


# 7) Hybrid - differential evolution + L-BFGS-B
res_de_lbfgsb = minimize(
    loss_function,
    x0=res_de.x, #The results from the differential evolution are used as the initial guess for a Bounded Quasi-Newton optimisation
    args=(df_calib_crra,),
    method="L-BFGS-B", 
    bounds=[(100, 30000), (0.5, 6.0)])

alpha_de_lbfgsb, sigma_de_lbfgsb = res_de_lbfgsb.x

# 8) Nelder-Mead random restarts
np.random.seed(123) #Random paths taken from the same starting point
n_starts = 50 #Restarts will happen 50x
best_nm_random_loss = np.inf
res_nm_random_best = None
best_nm_start_alpha = None
best_nm_start_sigma = None

for i in range(n_starts):
    alpha_start = np.random.uniform(500, 20000) #Chooses any random number for alphs in those bounds
    sigma_start = np.random.uniform(0.5, 6) #Chooses any random value for sigma in those bounds
    res_temp = minimize(
        loss_function,
        x0=[alpha_start, sigma_start], #Initial guess is the random number from above
        args=(df_calib_crra,), 
        method="Nelder-Mead",
        options={
            "maxiter": 5000, #Operation will stop after 5000 steps 
            "xatol": 1e-6, #This is the tolerance saying to not change alpha if less than this value 
            "fatol": 1e-6}) #Means stop the loop if tolerence of the loss fucntion if less than this valus

    if res_temp.fun < best_nm_random_loss: #Constantly comparing currently best loss value to the next loss value
        best_nm_random_loss = res_temp.fun
        res_nm_random_best = res_temp
        best_nm_start_alpha = alpha_start
        best_nm_start_sigma = sigma_start

alpha_nm_random, sigma_nm_random = res_nm_random_best.x

#Containing all the results in a variable so they can be easily converted to a Pandads Table
results = [{
        "Method": "L-BFGS-B",
        "Alpha": alpha_lbfgsb,
        "Sigma": sigma_lbfgsb,
        "Loss": res_lbfgsb.fun},

    {"Method": "Nelder-Mead (single)",
        "Alpha": alpha_nm_single,
        "Sigma": sigma_nm_single,
        "Loss": res_nm_single.fun},

    {   "Method": "BFGS",
        "Alpha": alpha_bfgs,
        "Sigma": sigma_bfgs,
        "Loss": res_bfgs.fun},

    {
        "Method": "Grid Search",
        "Alpha": best_grid_alpha,
        "Sigma": best_grid_sigma,
        "Loss": best_grid_loss},

    {   "Method": "Grid + BFGS",
        "Alpha": alpha_grid_bfgs,
        "Sigma": sigma_grid_bfgs,
        "Loss": res_grid_bfgs.fun},

    {   "Method": "Differential Evolution",
        "Alpha": alpha_de,
        "Sigma": sigma_de,
        "Loss": res_de.fun},

    {   "Method": "DE + L-BFGS-B",
        "Alpha": alpha_de_lbfgsb,
        "Sigma": sigma_de_lbfgsb,
        "Loss": res_de_lbfgsb.fun},

    {
        "Method": "Nelder-Mead (random restarts)",
        "Alpha": alpha_nm_random,
        "Sigma": sigma_nm_random,
        "Loss": best_nm_random_loss}]


#Converting results variable to Pandas DataFrame
df_results = pd.DataFrame(results)

#Rounding all results for presentation
df_results[["Alpha", "Sigma", "Loss"]] = df_results[["Alpha", "Sigma", "Loss"]].round(4)
latex_table = df_results.to_latex(index=False, float_format="%.4f")
print(latex_table)
print(df_results)




#-------------------------------------------REPRODUCING TABLE WITH ALPHA & SIGMA CALIBRATED-----------------------------------------




#Creating a new data frame for the jointly calibrated variables to be presented in a table
df_jointly_calibrated_table = df_calib_crra.copy()

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
    lambda row: solve_h_brentq(row["tau"], row["c_y"], alpha_de_lbfgsb, sigma_de_lbfgsb),
    axis=1)

#Adding in  a difference column
df_jointly_calibrated_table["Difference"] = df_jointly_calibrated_table["Predict"] - df_jointly_calibrated_table["Actual"]

#Round new columns to 1dp to match the rest of the table
df_jointly_calibrated_table["Predict"] = df_jointly_calibrated_table["Predict"].round(1)
df_jointly_calibrated_table["Difference"] = df_jointly_calibrated_table["Difference"].round(1)

print(df_jointly_calibrated_table)

#Converting pandas table to latex so it can be easily used in document 
latex_table_1a = df_jointly_calibrated_table.to_latex(index = False, float_format = "%.1f") #Rounds to 1dp
print(latex_table_1a)



#-------------------------------------------------------PLOTTING PREDICTED VS CALIBRATED VALUES---------------------------------------------



#We are going to be using ax rather than plt to explicitly seperate axes from different diagrams to avoid potential syntax errors

#Plotting data for 1993 - 1996 first
fig, ax = plt.subplots() #This is creating a set of axis and a figure

#Changing labels back to abbreviated form for the plotting to avoid cluttering the figure
labels = {
    "Germany": "DEU",
    "France": "FRA",
    "Italy": "ITA",
    "Canada": "CAN",
    "United Kingdom": "GBR",
    "Japan": "JPN",
    "United States": "USA"}

#Using df_table rather than df_calib since it has all updated columns
df_1993 = df_jointly_calibrated_table[df_jointly_calibrated_table["Period"] == "1993-96"].copy() # .copy means we dont make changes to original data frame
df_1993["label"] = df_1993["Country"].map(labels)

print("\n",df_1993)
ax.scatter(df_1993["Actual"], df_1993["Predict"], color = "red", s = 30)

#Adding in the country abbreviations to the scatter plot
for _, row in df_1993.iterrows():
    ax.text(
        row["Actual"] + 0.1,   #Shifting actual point slightly right to avoid a messy graph
        row["Predict"] + 0.1,  #Shift predicted point up slightly
        row["label"],          #Adding the label to plot
        fontsize = 10)

#Adding in the 45 degree dashed line as shown in the lecture slides
xmin = min(df_1993["Actual"].min(), df_1993["Predict"].min()) - 1 #Finding smallest value across both axes
xmax = max(df_1993["Actual"].max(), df_1993["Predict"].max()) + 1 #Finding largest value across both axes

ax.plot([xmin, xmax], [xmin, xmax], linestyle = "--", color = "black") #Plotting the dashed line


#Limiting both axes so they match that of the lecture slides
ax.set_xlim(16, 28)
ax.set_ylim(16, 28)

ax.set_title("Model Fit: 1993-96")
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")

fig.tight_layout()
fig.savefig("Model Fit: 1993-96.pdf", dpi = 300, bbox_inches = "tight")
plt.show()




#Plotting data for 1970 - 1974
fig, ax = plt.subplots() #This is creating a set of axis and a figure

#Using df_jointly_calibrated_table rather than df_calib since it has all updated columns
df_1970 = df_jointly_calibrated_table[df_jointly_calibrated_table["Period"] == "1970-74"].copy() # .copy means we dont make changes to original data frame
df_1970["label"] = df_1970["Country"].map(labels)

ax.scatter(df_1970["Actual"], df_1970["Predict"], color = "red", s = 30)

#Adding in the country abbreviations to the scatter plot
for _, row in df_1970.iterrows():
    ax.text(
        row["Actual"] + 0.1,   #Shifting actual point slightly right to avoid a messy graph
        row["Predict"] + 0.1,  #Shift predicted point up slightly
        row["label"],          #Adding the label to plot
        fontsize = 10)

#Adding in the 45 degree dashed line
xmin = min(df_1970["Actual"].min(), df_1970["Predict"].min()) - 1 #Finding smallest value across both axes
xmax = max(df_1970["Actual"].max(), df_1970["Predict"].max()) + 1 #Finding largest value across both axes

ax.plot([xmin, xmax], [xmin, xmax], linestyle = "--", color = "black")


#Limiting both axes so they match that of the lecture slides
ax.set_xlim(18, 30)
ax.set_ylim(18, 34)

ax.set_title("CRRA Model Fit: 1970-74")
ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")

fig.tight_layout()
fig.savefig("Model Fit: 1970-74.pdf", dpi = 300, bbox_inches = "tight")
plt.show()



#----------------------------------------------CREATING AN IMPROVEMENT COLUMN FOR PRESCOTT AND JOINT CALIBRATED MODEL----------------------------------------------------



#This code is coppied from Q1 script as it uses the exact same data with the exception of changing the Pandas DataFrame name 

#Prescott data using the clean data frame which does not include the CRRA utilty function
df_prescott = df_clean.copy()

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
    lambda row: prescott_foc_equilibrium_condition(row["tau"], row["c_y"]), axis=1)

#Adding in the difference column
df_prescott["Difference"] = df_prescott["Predict"] - df_prescott["Actual"]

#Rounding results for presentation
df_prescott["Predict"] = df_prescott["Predict"].round(1)
df_prescott["Difference"] = df_prescott["Difference"].round(1)



#Creating a new data frame to compare the data from Prescotts model and the CRRA model where parameters were jointly calibrated
df_compare = df_jointly_calibrated_table.copy()

df_compare["Prescott Difference"] = df_prescott["Difference"]
df_compare["CRRA Difference"] = df_jointly_calibrated_table["Difference"]

#Calculating the difference between the Prescott data and the CRRA data
df_compare["Improvement"] = df_compare["Prescott Difference"].abs() - df_compare["CRRA Difference"].abs()

#Creating the columns for the new table comparing the values
df_compare = df_compare[["Period", "Country", "Prescott Difference", "CRRA Difference", "Improvement"]]

print(df_compare.to_latex(index=False, float_format="%.1f")) #Latex form 
print("\n",df_compare)




#---------------------------------------------GRAPHING THE SURFACE AND CONTOUR OF THE LOSS FUNCTION---------------------------------------


#Plotting the ranges used for both surface and contour
alpha_plot_range = (2000, 20000, 150)
sigma_plot_range = (2.0, 4.0, 150)


#Alphas and sigmas are looped into the loss function and the loss value is stored which computes the plotted surface 
def plot_loss_surface(df, loss_function, alpha_range, sigma_range, optimiser_points = None):
    
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
    ax.legend()

    fig.tight_layout()
    fig.savefig("Loss surface.pdf", dpi = 300, bbox_inches = "tight", pad_inches = 0.3)
    plt.show()

    return A, S, Z

#Allows for presentational changes to be made outside the plotting function to avoid unnecessary syntaxs
optimiser_points = { #Change parameters below for different visualisation for both graphs
    "L-BFGS-B (Bounded Quasi-Newton)": {
            "alpha": alpha_lbfgsb,
            "sigma": sigma_lbfgsb,
            "colour": "blue",
            "size": 120,
            "marker": "^"},


    "Nelder-Mead Single": {
            "alpha": alpha_nm_single,
            "sigma": sigma_nm_single,
            "colour": "pink",
            "size": 120,
            "marker": "o"},

    "BFGS (Quasi-Newton)": {
            "alpha": alpha_bfgs,
            "sigma": sigma_bfgs,
            "colour": "black",
            "size": 120,
            "marker": "X"},

    "Nelder-Mead Random Restarts": {
            "alpha": alpha_nm_random,
            "sigma": sigma_nm_random,
            "colour": "brown",
            "size": 120,
            "marker": "d"},

    "Grid Search + Quasi-Newton": {
            "alpha": alpha_grid_bfgs,
            "sigma": sigma_grid_bfgs,
            "colour": "orange",
            "size": 120,
            "marker": "s"},

    "Grid Search": {
            "alpha": best_grid_alpha,
            "sigma": best_grid_sigma,
            "colour": "green",
            "size": 120,
            "marker": "v"},

    "Differential Evolution": {
            "alpha": alpha_de,
            "sigma": sigma_de,
            "colour": "gray",
            "size": 120,
            "marker": "*"},


    "Differential Evolution + Bounded Quasi-Newton": {
            "alpha": alpha_de_lbfgsb,
            "sigma": sigma_de_lbfgsb,
            "colour": "red",
            "size": 120,
            "marker": "v"}}


#Below is the line which calls the function and plots the loss surface
A, S, Z = plot_loss_surface(df_calib_crra, loss_function, alpha_plot_range, sigma_plot_range, optimiser_points)



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

ax.set_xlim(10000, 20000) #Limiting the axis for better clarity of results
ax.set_ylim(2.95, 3.1)

ax.set_xlabel("Alpha")
ax.set_ylabel("Sigma")

ax.set_title("Loss Function Contour")
ax.legend()

fig.tight_layout()
fig.savefig("Contour of loss surface.pdf", dpi = 300, bbox_inches = "tight", pad_inches = 0.5)

plt.show()