import polars as pl
import numpy as np
from scipy.optimize import minimize_scalar, minimize
import matplotlib.pyplot as plt
from pathlib import Path

# set up output path globals 
DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

def load_g7_data():
    """
    Reads all raw data files, transforms, renames and returns single wide frame
    Countries down rows, variables along columns

    Country Names use:
    Canada
    France
    Germany
    Italy
    Japan
    United Kingdom
    United States
    """

    RENAME_DICT_GDP = {
        "Country or Area": "country",
        "General government final consumption expenditure": "G",
        "Gross capital formation": "I",
        "Gross Domestic Product (GDP)": "GDP",
        "Year" : "year"
        }
    
    RENAME_DICT_HH = {
        "Country or Area": "country",
        "Current taxes on income, wealth, etc.": "DT",
        "Social contributions" : "SST"
    }
    
    DEF_GDP_PERC = 0.012845 # Canada defence expenditure as % of GDP - taken from World Bank data

    # UN SNA data
    df_gdp = pl.read_csv(DATA_DIR / "un_gdp.csv")
    df_con = pl.read_csv(DATA_DIR / "un_consumption.csv")
    df_gov_cofog = pl.read_csv(DATA_DIR / "un_gov_cofog.csv")
    df_ind_tax = pl.read_csv(DATA_DIR / "un_indirect_tax.csv")
    df_econ = pl.read_csv(DATA_DIR / "un_total_economy.csv", schema_overrides={"Value": pl.Float64, "SNA93 Table Code": pl.Utf8})
    df_hh = pl.read_csv(DATA_DIR / "un_household.csv", schema_overrides={"SNA93 Table Code": pl.Utf8})

    # OECD data
    df_ldpr = pl.read_csv(DATA_DIR / "labour_force_rate.csv", skip_rows = 2)
    df_hours = pl.read_csv(DATA_DIR / "hours_worked.csv", skip_rows = 2)
    df_tax_wedge = pl.read_csv(DATA_DIR / "tax_wedge.csv", skip_rows = 2)

    # replace UK name in gpd file to be consistent with others
    df_gdp = df_gdp.with_columns(
    pl.col("Country or Area")
    .replace({"United Kingdom of Great Britain and Northern Ireland": "United Kingdom"})
    .alias("Country or Area"))

    # start with gdp as this has 3 of the values we want (g, i, gdp)
    df_wide = df_gdp.pivot(values = ["Value"], index = ["Country or Area", "Year"], on = "Item")
    df_wide = df_wide.rename(RENAME_DICT_GDP)

    # next we take consumption from the con dataframe (it's included in GDP but that has NPISH that we want to exclude)
    df_c = df_con.filter(pl.col("Item") == "Equals: Household final consumption expenditure").select(["Country or Area", "Value"]).rename({"Value" : "C", "Country or Area": "country"}) # rename Value here or it will clash when merging

    df_merged = df_wide.join(df_c, on = ["country"], how = "left")

    #now we want to get the millitary expenditure portion of governemnt consumption
    df_mil = df_gov_cofog.filter(pl.col("Item") == "Defence").select(["Country or Area", "Value"]).rename({"Value" : "G_mil", "Country or Area": "country"})

    df_merged = df_merged.join(df_mil, on = ["country"], how = "left")

    # since Canada doesn't have government defence spending in the dataset we calculate is separately
    df_merged = df_merged.with_columns(
            G_mil = pl.when(pl.col("country") == "Canada") # create a new column to replace the existing one. Only change value for Canada's row
            .then(pl.col("GDP") * DEF_GDP_PERC)
            .otherwise(pl.col("G_mil")))

    # prepare indirect tax df for merge
    df_it = df_ind_tax.select([pl.col("Country or Area").alias("country"), pl.col("Value").alias("IT")])

    df_merged = df_merged.join(df_it, on = ["country"], how = "left")

    # transform household df ready for merge

    df_hh_clean = df_hh.filter(
        pl.col("Item").is_in([
            "Current taxes on income, wealth, etc.",
            "Social contributions",
            ])
            ).select(["Country or Area", "Item", "Value"])

    df_hh_wide = df_hh_clean.pivot(values = ["Value"], index = ["Country or Area"], on = "Item")
    df_hh_wide = df_hh_wide.rename(RENAME_DICT_HH)

    df_merged = df_merged.join(df_hh_wide, on = "country", how = "left")

    # add depreciation from economy level dataframe
    df_dep = (df_econ.filter(
        pl.col("Item").is_in(["Less: Consumption of fixed capital"]))
        .select(["Country or Area","Value"])
        .rename({"Value" : "dep", "Country or Area": "country"}))
    
    df_merged = df_merged.join(df_dep, on = "country", how = "left")
    
    # OECD variables - no transformation needed before joining
    df_merged = df_merged.join(df_ldpr.rename({"Category" : "country", "25-64 year-olds": "lfpr"}), on="country", how="left")
    df_merged = df_merged.join(df_hours.rename({"Category" : "country", "Hours per year per person": "h_py"}), on="country", how="left")
    df_merged = df_merged.join(df_tax_wedge.rename({"Category" : "country", "Average tax wedge": "t_h_alt"}), on="country", how="left") # alternate measure using tax wedge

    # transform percentages to decimals after join
    df_merged = df_merged.with_columns([
        (pl.col("lfpr") / 100).alias("lfpr"),
        (pl.col("t_h_alt") / 100).alias("t_h_alt"),])

    return df_merged

def generate_model_data(df_elements, theta = 0.32):
    """
    Takes transformed raw data, creates all intermediate elements and final variables
    Returns elements and final dfs
    """

    # compute intermediate values
    df_elements = (df_elements.with_columns([
        ((2 / 3 + 1 / 3 * (pl.col("C") / (pl.col("C") + pl.col("I")))) * pl.col("IT")).alias("IT_c"),
        (pl.col("GDP") - pl.col("IT")).alias("y"),
        (pl.col("h_py") / 52).alias("h_pw"),]).with_columns([(pl.col("C") + pl.col("G") - pl.col("G_mil") - pl.col("IT_c")).alias("c"),
        (pl.col("IT_c") / (pl.col("C") - pl.col("IT_c"))).alias("t_c"),
        (pl.col("SST") / ((1 - theta) * (pl.col("GDP") - pl.col("IT")))).alias("t_ss"),
        (pl.col("DT") / (pl.col("GDP") - pl.col("IT") - pl.col("dep"))).alias("t_inc")
        ])
        .with_columns(
            (pl.col("t_ss") + 1.6 * pl.col("t_inc")).alias("t_h")
        )
    )
    

    # compute final values
    df_final = df_elements.select([
        (pl.col("country")),
        (pl.col("year")).cast(pl.String),
        (pl.col("h_pw") * pl.col("lfpr")).alias("h_obs"),
        ((pl.col("t_h") + pl.col("t_c")) / (1 + pl.col("t_c"))).alias("t"),
        ((pl.col("t_h_alt") + pl.col("t_c")) / (1 + pl.col("t_c"))).alias("t_alt"),
        (pl.col("c") / pl.col("y")).alias("c_y"),
    ])

    return df_final, df_elements

def validate_model_data(df_elements, df_final):
    """
    Run basic consistency checks on intermediate and final Prescott-style data.
    Raises AssertionError if any check fails.
    """

    assert df_elements.height == 7, "Expected 7 G7 countries."
    assert df_final.height == 7, "Expected 7 final rows."

    # Null checks
    for col in ["C", "G", "I", "GDP", "G_mil", "IT", "lfpr", "h_py", "t_h", "t_h_alt"]:
        assert df_elements.select(pl.col(col).is_null().sum()).item() == 0, f"Nulls found in {col}"

    for col in ["IT_c", "y", "h_pw", "c", "t_c"]:
        assert df_elements.select(pl.col(col).is_null().sum()).item() == 0, f"Nulls found in {col}"

    for col in ["h_obs", "t", "c_y"]:
        assert df_final.select(pl.col(col).is_null().sum()).item() == 0, f"Nulls found in {col}"

    # Range checks
    assert df_elements.filter((pl.col("lfpr") <= 0) | (pl.col("lfpr") >= 1)).height == 0, "lfpr must be in (0,1)"
    assert df_elements.filter((pl.col("t_h") < 0) | (pl.col("t_h") >= 1)).height == 0, "t_h must be in [0,1)"
    assert df_elements.filter(pl.col("IT") <= 0).height == 0, "IT must be positive"
    assert df_elements.filter(pl.col("GDP") <= 0).height == 0, "GDP must be positive"
    assert df_elements.filter(pl.col("C") <= 0).height == 0, "C must be positive"
    assert df_elements.filter(pl.col("I") < 0).height == 0, "I must be non-negative"
    assert df_elements.filter(pl.col("G") < 0).height == 0, "G must be non-negative"
    assert df_elements.filter((pl.col("G_mil") < 0) | (pl.col("G_mil") > pl.col("G"))).height == 0, "G_mil must be between 0 and G"

    # Constructed values
    assert df_elements.filter(pl.col("IT_c") <= 0).height == 0, "IT_c must be positive"
    assert df_elements.filter(pl.col("IT_c") >= pl.col("C")).height == 0, "IT_c must be less than C"
    assert df_elements.filter(pl.col("y") <= 0).height == 0, "y must be positive"
    assert df_elements.filter(pl.col("c") <= 0).height == 0, "c must be positive"
    assert df_elements.filter((pl.col("t_c") < 0) | (pl.col("t_c") >= 1)).height == 0, "t_c must be in [0,1)"

    # Final plausibility
    assert df_final.filter((pl.col("h_obs") <= 0) | (pl.col("h_obs") >= 100)).height == 0, "h_obs out of plausible weekly range"
    assert df_final.filter((pl.col("t") < 0) | (pl.col("t") >= 1)).height == 0, "t must be in [0,1)"
    assert df_final.filter((pl.col("c_y") <= 0) | (pl.col("c_y") >= 2)).height == 0, "c_y out of plausible range"

def run_data_pipeline():
    """
    Runs data import and model data genetaion.
    Creates output directory if non-existant and saves elements and final df within
    """

    # create output directory if it doesn't already exist
    OUTPUT_DIR.mkdir(exist_ok=True) 

    df_merged = load_g7_data()
    df_final, df_elements = generate_model_data(df_merged)

    # run validation to sense check all data transformations - if it fails it will raise an assertion error
    validate_model_data(df_elements, df_final)

    # save dataframes for use in report
    print("Saving elements csv...")
    df_elements.write_csv(OUTPUT_DIR / "df_elements.csv")
    print("Saving final csv...")
    df_final.write_csv(OUTPUT_DIR / "df_final.csv")

def h(theta: float, alpha: float, t, c_o): # return prescott model predicted hours can run vectorise or scalar

    num = 100 * (1 - theta)
    den = (alpha / (1 - t)) * c_o + 1 - theta

    return num / den

def calibrate_alpha(df, tax_col, alpha_guess = 1.54, theta = 0.32):
    """
    Uses 3 minimisation routines to minimise loss function over alpha
    returns dict containing miniser (alpha*) for each routine
    """

    results = {} # instantiate dict for results

    res_bounded = minimize_scalar(lambda alpha: loss_function(df, tax_col, alpha, theta), method = "bounded", bounds = (0, 100)) # large bound
    res_brent = minimize_scalar(lambda alpha: loss_function(df, tax_col, alpha, theta), method = "brent", bracket = (0, alpha_guess, 10)) # large bound 
    res_nm = minimize(lambda alpha: loss_function(df, tax_col, alpha, theta), x0 = [alpha_guess], method = "Nelder-Mead")

    # save results in dict
    results["bounded"] = res_bounded.x
    results["brent"] = res_brent.x
    results["nm"] = res_nm.x[0]

    return results

def loss_function(df, tax_col, alpha: float, theta: float) -> float:
    """
    Square error loss function over prescott dataframe
    returns scalar loss value
    """
    #elementwise loss
    h_pred = h(theta, alpha, df[tax_col], df["c_y"]) # compute predicted hours over dataframe
    loss = ((df["h_obs"] - h_pred) ** 2).sum() # loss = sum of MSE of series
    
    return float(loss)

def q2_b():

    alpha_guess = 1.54
    theta = 0.32
    tax_cols = ["t", "t_alt"] # Prescott measure and OECD tax wedge

    df_2019 = pl.read_csv(OUTPUT_DIR / "df_final.csv", schema_overrides={"year": pl.String}) # need to override year since polars infers it as int
    df_1994 = pl.read_csv(DATA_DIR / "prescott_94_96.csv")
    df_1994 = df_1994.with_columns(pl.col("t").alias("t_alt")) # copy t into the t_alt for later calibration
    
    
    # Ensure both have the same columns in the same order
    model_cols = ["country", "year", "h_obs", "c_y", "t", "t_alt"]
    df_2019 = df_2019.select(model_cols)
    df_1994 = df_1994.select(model_cols)

    df = pl.concat([df_2019, df_1994])

    calibration_rows = []

    for col in tax_cols:

        print(f"============= Results using {col} as the Tax column ============= \n")

        results = calibrate_alpha(df, col, alpha_guess, theta)

        for method, alpha in results.items():
            print(f"Search Method: {method}")
            print(f"Alpha: {alpha: .6f}")

            pred_col = f"h_pred_{method}_{col}"
            diff_col = f"diff_{method}_{col}"

            df = df.with_columns([h(theta, alpha, df[col], df["c_y"]).alias(pred_col),
            ]).with_columns([(pl.col(pred_col) - pl.col("h_obs")).alias(diff_col),])

            diff_2019 = df.filter(pl.col("year") == "2019")[diff_col].abs().sum()
            
            # Error for 1993-96
            diff_1994 = df.filter(pl.col("year") == "1993-96")[diff_col].abs().sum()

            print(f"Total Absolute Error (2019):    {diff_2019: .4f}")
            print(f"Total Absolute Error (1993-96): {diff_1994: .4f}")
            print(f"Total Absolute Error:           {diff_2019 + diff_1994: .4f}\n")

            calibration_rows.append({
                "tax_col": col,
                "method": method,
                "alpha": alpha
            })

    pl.DataFrame(calibration_rows).write_csv(OUTPUT_DIR / "calibration_results.csv")
    df.write_csv(OUTPUT_DIR / "model_predictions.csv")


def plot_predictions(save_path = None):
    method = "nm" # lowest loss method

    df = pl.read_csv(OUTPUT_DIR / "model_predictions.csv", schema_overrides={"year": pl.String})
    periods = ["1993-96", "2019"]
    
    # Define the two prediction columns based on the results
    pred_t = f"h_pred_{method}_t"
    pred_t_alt = f"h_pred_{method}_t_alt"

    # We check h_obs and BOTH prediction columns to find the chart boundaries
    relevant_data = df.filter(pl.col("year").is_in(periods))
    all_values = (
        relevant_data["h_obs"].to_list() + 
        relevant_data[pred_t].to_list() + 
        relevant_data[pred_t_alt].to_list())
    
    global_min = min(all_values) - 0.5
    global_max = max(all_values) + 0.5

    for period in periods:
        df_period = df.filter(pl.col("year") == period)

        x = df_period["h_obs"].to_list()
        y_t = df_period[pred_t].to_list()
        y_alt = df_period[pred_t_alt].to_list()
        countries = df_period["country"].to_list()

        fig, ax = plt.subplots(figsize=(9, 7))

        # Plot Prescott predictions in Red
        ax.scatter(x, y_t, color="red", label="Prescott Method (t)", zorder=3)
        
        # Plot OECD predictions in Blue
        ax.scatter(x, y_alt, color="blue", marker="s", label="OECD Tax Wedge (t_alt)", zorder=3)

        # 45-degree line (Perfect Prediction Line)
        ax.plot([global_min, global_max], [global_min, global_max], 
                linestyle="--", color="black", alpha=0.5, label="Perfect Fit (45°)")

        # Country labels - we'll attach them to the observed x-value
        for i, label in enumerate(countries):
            ax.annotate(label, (x[i], y_t[i]), xytext=(5, 5), 
                        textcoords="offset points", fontsize=12, alpha=0.7)

        ax.set_title(f"Comparison of Tax Models: Predicted vs Observed Hours ({period})")
        ax.set_xlabel("Observed Labour Supply (Hours per week)")
        ax.set_ylabel("Predicted Labour Supply (Hours per week)")
        
        ax.legend(loc="upper left")

        # Global Limits & Grid 
        ax.set_xlim(global_min, global_max)
        ax.set_ylim(global_min, global_max)
        ax.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()

        if save_path is not None:
            safe_period = str(period).replace("/", "-").replace(" ", "_")
            plt.savefig(f"{save_path}_comparison_{safe_period}.pdf", dpi=300, bbox_inches="tight")
            plt.close()
        else:
            plt.show()

def main():

    no_final_data = False # if final data hasn't been created set to True

    if no_final_data == True: # only creates final data if True
        run_data_pipeline()

    q2_b() # needs outputs/df_final.csv to run
    plot_predictions(save_path = OUTPUT_DIR / "diff_fig")

if __name__ == "__main__":
    main()