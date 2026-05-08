import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
import matplotlib.pyplot as plt

# ---------------------------------------------IMPORTING AND CLEANING THE DATA-------------------------------------------------

#Importing Prescott data 
DATA_PATH = "data/data_prescott.csv"

df_raw = pd.read_csv(DATA_PATH)

df_clean = df_raw.copy() #Creating a new copy of the raw data to clean labelled clean

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


#--------------------------------------------------------------------CALIBRATING ALPHA-----------------------------------------------------------------

df_calib = df_clean.copy()

#Defining equilibrium condition function with  initial guess for alpha
def equilibrium_condition_calib(tau, c_y, theta = 0.32, alpha_guess = 2.75):
     return 100 * (1 - theta) / ((alpha_guess * c_y / (1 - tau)) + (1 - theta))


#Creating the loss function for alpha to be calibrated on
def loss_function(df, alpha, theta):
    h_pred = equilibrium_condition_calib(df["t"], df["c_y"], theta, alpha)
    loss = ((df["h_obs"] - h_pred) ** 2).sum()
    return float(loss)


#Calibrating alpha using 3 different methods, bounded, brent and nelder-mead
def calibrate_alpha(df, alpha_guess = 1.54, theta = 0.32):

    results = {} #Empty container for results to be stored in

    res_bounded = minimize_scalar(
        lambda alpha: loss_function(df, float(alpha), theta), #float means we expect alpha to be a decimal
        method = "bounded", #Lambda is creating a temporary loss fucntion defined above
        bounds = (0.1, 5))

    res_brent = minimize_scalar(
        lambda alpha: loss_function(df, float(alpha), theta),
        method = "brent", 
        bracket = (0.1, alpha_guess, 5))

    res_nm = minimize(
        lambda alpha: loss_function(df, float(alpha[0]), theta),
        x0 = [alpha_guess], #Uses the guess for alpha hard coded within the function
        method = "Nelder-Mead")

    results["bounded"] = res_bounded.x #These lines are filling in the results found from the optimisers into the empty container
    results["brent"] = res_brent.x
    results["nm"] = res_nm.x[0]

    return results


results = calibrate_alpha(df_calib)

#Extracting alpha values from array and printing them
print("\nCalibrated Alpha Values:")
print(f"Bounded: {results['bounded']:.4f}")
print(f"Brent: {results['brent']:.4f}")
print(f"Nelder-Mead: {results['nm']:.4f}\n")



#-------------------------------------------REPRODUCING TABLE WITH ALPHA CALIBRATED-----------------------------------------




#Creating the pandas data frame so it looks like the one on the lecture slides
df_prescott = df_calib.copy()


#Defining equilibrium condition function with the calibrated alpha value found above
def equilibrium_condition(tau, c_y, theta = 0.32, alpha = 1.7105):
     return 100 * (1 - theta) / ((alpha * c_y / (1 - tau)) + (1 - theta))


#Renaming columns to match lecture slides
df_prescott = df_prescott.rename(columns = {
    "period": "Period",
    "country": "Country",
    "h_obs": "Actual",
    "t": "tau",
    "c_y": "c_y"})

#Making years to match the lecture slides
df_prescott["Period"] = df_prescott["Period"].replace({
    "1993-1996": "1993-96",
    "1970-1974": "1970-74"})

#Keeping only the variables listed below in the table 
df_prescott = df_prescott[["Period", "Country", "Actual", "tau", "c_y"]]

#Adding predicted labour supply using Q1 calibrated alpha
df_prescott["Predict"] = df_prescott.apply(
    lambda row: equilibrium_condition(row["tau"], row["c_y"]),
    axis=1)

#Adding in a difference column
df_prescott["Difference"] = df_prescott["Predict"] - df_prescott["Actual"]

#Rounding results to match the rest of the table
df_prescott["Predict"] = df_prescott["Predict"].round(1)
df_prescott["Difference"] = df_prescott["Difference"].round(1)

print(df_prescott)

#Converting pandas table to latex so it can be easily used in document 
latex_table_1a = df_prescott.to_latex(index = False, float_format = "%.1f") #Rounds to 1dp
print(latex_table_1a)
print("\n",df_prescott[["Period", "Country", "Actual", "tau", "c_y", "Predict", "Difference"]])



#-------------------------------------------------------PLOTTING PREDICTED VS CALIBRATED VALUES---------------------------------------------



#We are going to be using ax rather than plt to explicitly seperate axes from different diagrams to avoid potential syntax errors


#Plotting the data for 1970 - 1973 first
fig, ax = plt.subplots() #This is creating a set of axis and a figure

#Next we are creating abbreviations for all the country names so they can be labelled on the scatter plot
labels = {
    "Germany": "DEU",
    "France": "FRA",
    "Italy": "ITA",
    "Canada": "CAN",
    "United Kingdom": "GBR",
    "Japan": "JPN",
    "United States": "USA"}

df_1970 = df_prescott[df_prescott["Period"] == "1970-74"].copy() #This means we are only using the years 1970 - 1974
df_1970["label"] = df_1970["Country"].map(labels)

ax.scatter(df_1970["Actual"], df_1970["Predict"], s = 20, color = "red") #This is adding the scatter points to the figure created above

#Adding in the country abbreviations to the scatter plot
for _, row in df_1970.iterrows():
    ax.text(row["Actual"] + 0.1, row["Predict"] + 0.1, row["label"], fontsize=11, color = "black") #This line is moving the labels slightly so more visible

#Creating the 45-degree line as shown in the lecture slides
xmin = min(df_1970["Actual"].min(), df_1970["Predict"].min()) - 1
xmax = max(df_1970["Actual"].max(), df_1970["Predict"].max()) + 1

ax.plot([xmin, xmax], [xmin, xmax], linestyle="--", color = "black") #Plotting the dashed line 

ax.set_xlim(18,30) #limiting both axis to macth lecture slides
ax.set_ylim(18,34)

ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title("Prescott Model Fit: 1970-74")

fig.tight_layout()
fig.savefig("Prescott Model: fit_1970_74.pdf", dpi=300, bbox_inches="tight") #Saves image so it can be used in text document
plt.show()



#Plotting the data for 1990 - 1993
fig, ax = plt.subplots() #This is creating a set of axis and a figure


df_1993 = df_prescott[df_prescott["Period"] == "1993-96"].copy() #This means we are only using the years 1993 - 1996
df_1993["label"] = df_1993["Country"].map(labels)

ax.scatter(df_1993["Actual"], df_1993["Predict"], s = 20, color = "red") #This is adding the scatter points to the figure created above

#Adding in the country abbreviations to the scatter plot
for _, row in df_1993.iterrows():
    ax.text(row["Actual"] + 0.1, row["Predict"] + 0.1, row["label"], fontsize=11, color = "black") #This line is moving the labels slightly so more visible

#Creating the 45-degree line as shown in the lecture slides
xmin = min(df_1993["Actual"].min(), df_1993["Predict"].min()) - 1
xmax = max(df_1993["Actual"].max(), df_1993["Predict"].max()) + 1

ax.plot([xmin, xmax], [xmin, xmax], linestyle="--", color = "black") #Plotting on the dashed line

ax.set_xlim(16,28) #limiting both axis to macth lecture slides
ax.set_ylim(16,28)

ax.set_xlabel("Actual")
ax.set_ylabel("Predicted")
ax.set_title("Prescott Model Fit: 1993-96")

fig.tight_layout()
fig.savefig("Prescott Model: fit_1993_96.pdf", dpi=300, bbox_inches="tight") #Saves image so it can be used in text document
plt.show()





# ---------------------------------------------------------COMPUTING THE CEV-------------------------------------------------------------------




#Defining the CEV function as seen in the lecture slides
def cev(hA, hB, alpha):
    return np.exp(np.log(hB / hA) + alpha * np.log((100 - hB) / (100 - hA))) - 1

def counterfactual_analysis(df_calib, tau_uk = 0.44, c_y_uk = 0.83, alpha = 1.7105, theta = 0.32):
    hA = equilibrium_condition(tau_uk, c_y_uk, theta, alpha) #Setting the UK's tax rate as the baseline

    #Creating the table as seen as in the brief
    results = pd.DataFrame({"Alternative Tax System": df_calib["country"], "tau": df_calib["t"]})
    
    #Creating the column counterfactual labour supply
    results["Counterfactual Labour Supply (hours)"] = results["tau"].apply(
        lambda tau: equilibrium_condition(tau, c_y_uk, theta, alpha)) #Each countries tax rate is plugged into FOC for the labour supply
    
    #Creating the column for welfare
    results["Welfare Change (epsilon)"] = results["Counterfactual Labour Supply (hours)"].apply(
        lambda hB: cev(hA, hB, alpha)) #We are now comparing baselines (UK tax rate) to the counterfactuals
    
    return results



#We use the df_clean as it already has full country names in the DataFrame

#Since we take the UK economy as a baseline, we must exclude it from the dataframe, which is what we are doing below
df_1993 = df_clean[df_calib["period"] == "1993-1996"].copy() #Using only the years 1993 - 1996
df_1993 = df_1993[df_1993["country"] != "United Kingdom"].copy() #The ! is removing the UK row from the data set 

results_cev = counterfactual_analysis(df_1993) #This line is taking the new  data set excluding the UK and running it through the function above

print("\n",results_cev)

results_b_latex = results_cev.to_latex(index=False, float_format="%.3f") #Converting the data table into latex form


print("\n", df_calib)