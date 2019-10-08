import sys
import pandas as pd
from functools import partial


dtype_dict = {
    "Variable": str,
    "Country": str,
    "Unit": str,
    "Year": int,
    "Value": float,
}

read_csv = partial(pd.read_csv, sep="\t", dtype=dtype_dict, index_col=False)


def combine_data(outputFile):
    fao_df = read_csv("data/fao_data.tsv")
    fao_df["Source"] = "FAO"

    wdi_df = read_csv("data/wdi_data.tsv")
    wdi_df["Source"] = "WDI"

    wdi_df["Unit"] = (
        wdi_df["Variable"].str.partition("(")[2].str.partition(")")[0]
    )
    wdi_df["Variable"] = wdi_df["Variable"].str.partition("(")[0]
    wdi_df = wdi_df.set_index(["Variable", "Unit", "Source", "Country"])
    fao_df = (
        fao_df.pivot_table(
            values="Value",
            index=["Variable", "Unit", "Source", "Country"],
            columns="Year",
        )
        .reset_index()
        .set_index(["Variable", "Unit", "Source", "Country"])
    )

    ind_cols = ["Variable", "Unit", "Source", "Country"]
    fao_wdi_df = pd.concat([fao_df, wdi_df], sort=True)

    fao_wdi_df = (
        fao_wdi_df.reset_index()
        .melt(id_vars=ind_cols, var_name="Year", value_name="Value")
        .dropna(subset=["Value"])
    )

    # If a column name is something like 2010-2012, we make copies of its data
    # for three years - 2010, 2011, 2012

    for c in fao_wdi_df.columns:
        if isinstance(c, str):
            if "-" in c:
                print(c)
                years = c.split("-")
                for y in range(int(years[0]), int(years[-1]) + 1):
                    y = str(y)
                    fao_wdi_df[y] = fao_wdi_df[y].fillna(fao_wdi_df[c])
                del fao_wdi_df[c]

    fao_wdi_df["State"] = None
    conflict_data_df = pd.read_csv(
        "data/raw/wm_12_month_evaluation/south_sudan_data_conflict.tsv",
        index_col=False,
        sep="\t",
        dtype=dtype_dict,
    )
    fewsnet_df = read_csv("data/indicator_data_fewsnet.tsv")
    fewsnet_df = fewsnet_df[(fewsnet_df.Value <= 5) & (fewsnet_df.Value >= 1)]

    climis_unicef_ieconomics_df = read_csv(
        "data/indicator_data_climis_unicef_ieconomics.tsv", thousands=","
    )

    dssat_df = read_csv("data/indicator_data_dssat.tsv")

    unhcr_df = read_csv("data/indicator_data_UNHCR.tsv")

    migration1_df = read_csv("data/south_sudan_migration_data_secondary.tsv")

    WHO1_df = pd.read_csv(
        "data/WHO-data1.csv", index_col=False, dtype=dtype_dict
    )
    WHO2_df = pd.read_csv(
        "data/WHO-data2.csv", index_col=False, dtype=dtype_dict
    )
    WHO3_df = pd.read_csv(
        "data/WHO-data3.csv", index_col=False, dtype=dtype_dict
    )

    IMF_df = pd.read_csv(
        "data/IMF-data.csv", index_col=False, dtype=dtype_dict
    )

    WFP_df = pd.read_csv(
        "data/WFP-data.csv", index_col=False, dtype=dtype_dict
    )

    acled1_df = pd.read_csv(
        "data/acled-data1.csv", index_col=False, dtype=dtype_dict
    )
    acled2_df = pd.read_csv(
        "data/acled-data2.csv", index_col=False, dtype=dtype_dict
    )
    acled3_df = pd.read_csv(
        "data/acled-data3.csv", index_col=False, dtype=dtype_dict
    )

    World_Bank_df = pd.read_csv(
        "data/World-Bank-data.csv", index_col=False, dtype=dtype_dict
    )

    dssat_oct2019_eval_data_df = read_csv('data/dssat_data_oct2019_eval.tsv')
    
    IOM_DTM1_df = pd.read_csv(
        "data/IOM-DTM-data1.csv", index_col=False, dtype=dtype_dict
    )
    
    combined_df = pd.concat(
        [
            fao_wdi_df,
            conflict_data_df,
            fewsnet_df,
            climis_unicef_ieconomics_df,
            dssat_df,
            unhcr_df,
            migration1_df,
            WHO1_df,
            WHO2_df,
            WHO3_df,
            IMF_df,
            WFP_df,
            acled1_df,
            acled2_df,
            acled3_df,
            World_Bank_df,
            dssat_oct2019_eval_data_df,
            IOM_DTM1_df,
        ],
        sort=True,
    ).dropna(subset=["Value"])
    combined_df.Variable = combined_df.Variable.str.strip()
    combined_df.to_csv(outputFile, sep="\t", index=False)
    with open("data/indicator_flat_list.txt", "w") as f:
        f.write("\n".join(set(combined_df.Variable)))


if __name__ == "__main__":
    combine_data(sys.argv[1])
