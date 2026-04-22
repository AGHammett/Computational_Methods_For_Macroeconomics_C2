import polars as pl
import numpy as np
from scipy.optimize import minimize_scalar
from q1 import h



def load_g7_data():
    """
    Country Names use:
    Canada
    France
    Germany
    Italy
    Japan
    United Kingdom
    United States
    """

    RENAME_DICT = {
        "Country or Area": "country",
        "General government final consumption expenditure": "G",
        "Gross capital formation": "I",
        "Gross Domestic Product (GDP)": "GDP",
        "Year" : "year"
        }
    
    DEF_GDP_PERC = 0.012845 # Canada defence expenditure as % of GDP - taken from World Bank data

    # UN SNA data
    df_gdp = pl.read_csv("q2data/un_gdp.csv")
    df_con = pl.read_csv("q2data/un_consumption.csv")
    df_gov_cofog = pl.read_csv("q2data/un_gov_cofog.csv")
    df_ind_tax = pl.read_csv("q2data/un_indirect_tax.csv")

    # OECD data
    df_ldpr = pl.read_csv("q2data/labour_force_rate.csv", skip_rows = 2)
    df_hours = pl.read_csv("q2data/hours_worked.csv", skip_rows = 2)
    df_tax_wedge = pl.read_csv("q2data/tax_wedge.csv", skip_rows = 2)

    # replace UK name in gpd file to be consistent with others
    df_gdp = df_gdp.with_columns(
    pl.col("Country or Area")
    .replace({"United Kingdom of Great Britain and Northern Ireland": "United Kingdom"})
    .alias("Country or Area"))

    # start with gdp as this has 3 of the values we want (g, i, gdp)
    df_wide = df_gdp.pivot(values = ["Value"], index = ["Country or Area", "Year"], on = "Item")
    df_wide = df_wide.rename(RENAME_DICT)

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
    
    # OECD variables - no transformation needed before joining
    df_merged = df_merged.join(df_ldpr.rename({"Category" : "country", "25-64 year-olds": "lfpr"}), on="country", how="left")
    df_merged = df_merged.join(df_hours.rename({"Category" : "country", "Hours per year per person": "h_py"}), on="country", how="left")
    df_merged = df_merged.join(df_tax_wedge.rename({"Category" : "country", "Average tax wedge": "t_h"}), on="country", how="left")

    # transform percentages to decimals after join
    df_merged = df_merged.with_columns([
        (pl.col("lfpr") / 100).alias("lfpr"),
        (pl.col("t_h") / 100).alias("t_h"),])

    return df_merged

def generate_model_data(df_elements):

    # compute intermediate values
    df_elements = df_elements.with_columns([
        ((2 / 3 + 1 / 3 * (pl.col("C") / (pl.col("C") + pl.col("I")))) * pl.col("IT")).alias("IT_c"),
        (pl.col("GDP") - pl.col("IT")).alias("y"),
        (pl.col("h_py") / 52).alias("h_pw"),]).with_columns([(pl.col("C") + pl.col("G") - pl.col("G_mil") - pl.col("IT_c")).alias("c"),
        (pl.col("IT_c") / (pl.col("C") - pl.col("IT_c"))).alias("t_c"),
        ])

    # compute final values
    df_final = df_elements.select([
        (pl.col("country")),
        (pl.col("year")),
        (pl.col("h_pw") * pl.col("lfpr")).alias("h_obs"),
        ((pl.col("t_h") + pl.col("t_c")) / (1 + pl.col("t_c"))).alias("t"),
        (pl.col("c") / pl.col("y")).alias("c_y"),
    ])

    return df_final, df_elements

def run_data_pipeline():

    df_merged = load_g7_data()
    df_final, df_elements = generate_model_data(df_merged)

    # run validation to sense check all data transformations - if it fails it will raise an assertion error
    validate_model_data(df_elements, df_final)

    # save dataframes for use in report
    df_elements.write_csv("outputs/df_elements.csv")
    df_final.write_csv("outputs/df_final.csv")

def validate_model_data(df_elements, df_final):
    """
    Run basic consistency checks on intermediate and final Prescott-style data.
    Raises AssertionError if any check fails.
    """

    assert df_elements.height == 7, "Expected 7 G7 countries."
    assert df_final.height == 7, "Expected 7 final rows."

    # Null checks
    for col in ["C", "G", "I", "GDP", "G_mil", "IT", "lfpr", "h_py", "t_h"]:
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

def calibrate_alpha(df, alpha_guess = 0, theta = 1.77):

    alpha = alpha_guess
    df["h_mod"] = h(theta, alpha, df["t"], df["c_y"])

    return df

def loss_function(h_obs: pl.Series, h_mod: pl.Series):
    """
    Arguments - h_obs, h_mod -> pandas columns
    
    Outputs - loss -> float (scalar)
    """
    #elementwise loss
    loss_series = (h_obs - h_mod)**2

    return float(sum(loss_series))

