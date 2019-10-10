import sys
import pandas as pd

def combine_data(outputFile):
    dtype_dict = {"Month": int, "Variable": str, "Country" : str, "Unit" : str, "Year":int, "Value": float}
    fao_df = pd.read_csv("data/fao_data.tsv", sep="\t", dtype=dtype_dict)
    fao_df["Source"] = "FAO"

    wdi_df = pd.read_csv("data/wdi_data.tsv", sep="\t", dtype=dtype_dict)
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
        if isinstance(c,str):
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
    )
    fewsnet_df = pd.read_csv(
        "data/indicator_data_fewsnet.tsv", index_col=False, sep="\t"
    )
    fewsnet_df = fewsnet_df[(fewsnet_df.Value <= 5) & (fewsnet_df.Value >= 1)]
    climis_unicef_ieconomics_df = pd.read_csv(
        "data/indicator_data_climis_unicef_ieconomics.tsv",
        index_col=False,
        sep="\t",
    )
    dssat_df = pd.read_csv(
        "data/indicator_data_dssat.tsv", index_col=False, sep="\t"
    )

    unhcr_df = pd.read_csv(
        "data/indicator_data_UNHCR.tsv", index_col=False, sep="\t"
    )

    migration1_df = pd.read_csv(
        "data/south_sudan_ReachJongleiJan_migration_data_old.tsv", index_col=False, sep="\t"
    )

    migration2_df = pd.read_csv(
        "data/south_sudan_54660_migration_data_old.tsv", index_col=False, sep="\t"
    )

    migration3_df = pd.read_csv(
        "data/south_sudan_62801_migration_data_old.tsv", index_col=False, sep="\t"
    )

    migration4_df = pd.read_csv(
        "data/south_sudan_62803_migration_data_old.tsv", index_col=False, sep="\t"
    )

    migration5_df = pd.read_csv(
        "data/south_sudan_63604_migration_data_old.tsv", index_col=False, sep="\t"
    )

    migration6_df = pd.read_csv(
        "data/south_sudan_UNHCR_migration_data_old.tsv", index_col=False, sep="\t"
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
            migration2_df,
            migration3_df,
            migration4_df,
            migration5_df,
            migration6_df,
        ],
        sort=True,
    ).dropna(subset=["Value"])
    combined_df.Variable = combined_df.Variable.str.strip()
    combined_df.to_csv(
        outputFile, sep="\t", index=False
    )
    with open("data/indicator_flat_list.txt", "w") as f:
        f.write("\n".join(set(combined_df.Variable)))

if __name__ == "__main__":
    combine_data(sys.argv[1])
