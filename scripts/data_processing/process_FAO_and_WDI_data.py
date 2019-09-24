import os, sys
import pandas as pd
from glob import glob
from future.utils import lmap
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from ruamel.yaml.scalarstring import DoubleQuotedScalarString as S
from ruamel.yaml import YAML
from delphi.utils.shell import cd
from delphi.utils.web import download_file
from pathlib import Path
import zipfile
import subprocess as sp
import contextlib


data_dir = Path("data")

def clean_FAOSTAT_data():

    faostat_dir = "data/raw/FAOSTAT"
    dfs = []
    for filename in tqdm(
        glob(str(faostat_dir) + "/*.csv"), desc="Processing FAOSTAT files"
    ):
        df = pd.read_csv(
            filename,
            encoding="latin-1",
            usecols=lambda colName: all(
                map(
                    lambda x: x not in colName,
                    ["Flag", "Note", "ElementGroup"],
                )
            ),
        dtype = {"Month": int, "State": str, "Country" : str, "Unit" : str, "Year":str, "Value": str}
        )

        df.rename(columns={"Country": "Area", "Months": "Month"}, inplace=True)
        if "Currency" in df.columns:
            df = df.rename(columns={"Currency": "Item", "Item": "Element"})
        if "Reporter Countries" in df.columns:
            df = df.rename(columns={"Reporter Countries": "Area"})
        if "Survey" in df.columns:
            df = df.rename(
                columns={
                    "Breadown by Sex of the Household Head": "Sex of the Household Head",
                    "Indicator": "Item",
                    "Measure": "Element",
                }
            )
            df["Area"] = df["Survey"].str.split().str.get(0)
            df["Year"] = df["Survey"].str.split().str.get(-1)
            del df["Survey"]
            df = df[df["Sex of the Household Head"] == "Total"]
            df = df[df["Breakdown Variable"] == "Country-level"]
            del df["Sex of the Household Head"]
            del df["Breakdown Variable"]
        if "Donor Country" in df.columns:
            df = df.rename(columns={"Recipient Country": "Area"})
            del df["Donor Country"]
            df["Element"] = "Food aid shipments"
        if set(df.columns.values) == {'Area', 'Area Code', 'Item','Item Code',  'Element', 'Element Code',
                'Year', 'Year Code', 'Value', 'Area Code', 'Unit', 'Value'}:
            df.Value = pd.to_numeric(df.Value, errors='coerce')
            df = df[df.Value.notnull()]
            df.rename(columns={"Area": "Country"}, inplace=True)
            df = df.query("Country == 'South Sudan' or Country == 'Ethiopia'")
            df = df[["Country", "Item", "Element", "Year", "Unit", "Value"]]
            df = df[~df.Year.str.contains('-')]
            df["Variable"] = df["Element"] + ", " + df["Item"]
            del df["Element"]
            del df["Item"]
            dfs.append(df)

    df = pd.concat(dfs)

    df.to_csv(
        str(data_dir / "fao_data.tsv"), index=False, sep="\t"
    )


def clean_WDI_data():
    print("Cleaning WDI data")
    df = pd.read_csv(
        "data/raw/WDI/WDIData.csv",
    )
    del df["Unnamed: 63"]
    df.rename(
        columns={"Indicator Name": "Variable", "Country Name": "Country"},
        inplace=True,
    )
    df.to_csv("data/wdi_data.tsv", index=False, sep="\t")


def combine_data():
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
    fao_df.dropna(subset=["Value"], inplace=True)
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
    fao_wdi_df = pd.concat([fao_df, wdi_df])


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
        "data/south_sudan_data_fewsnet.tsv", index_col=False, sep="\t"
    )
    fewsnet_df = fewsnet_df[(fewsnet_df.Value <= 5) & (fewsnet_df.Value >= 1)]
    climis_unicef_ieconomics_df = pd.read_csv(
        "data/south_sudan_data_climis_unicef_ieconomics.tsv",
        index_col=False,
        sep="\t",
    )
    dssat_df = pd.read_csv(
        "data/south_sudan_data_dssat.tsv", index_col=False, sep="\t"
    )

    unhcr_df = pd.read_csv(
        "data/south_sudan_data_UNHCR.tsv", index_col=False, sep="\t"
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
        Path(data_dir) / "indicator_data.tsv", sep="\t", index=False
    )
    with open("data/indicator_flat_list.txt", "w") as f:
        f.write("\n".join(set(combined_df.Variable)))


if __name__ == "__main__":
    if sys.argv[1] == "--fao":
        clean_FAOSTAT_data()
    elif sys.argv[1] == "--wdi":
        clean_WDI_data()
    elif sys.argv[1] == "--combine":
        combine_data()
