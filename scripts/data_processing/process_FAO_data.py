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

def clean_FAOSTAT_data(outputFile):

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
                    ["Code", "Flag", "Note", "ElementGroup"],
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
        if set(df.columns.values) == {'Area', 'Item','Element', 'Year', 'Value', 'Unit', 'Value'}:
            df.Value = pd.to_numeric(df.Value, errors='coerce')
            df = df[df.Value.notnull()]
            df.rename(columns={"Area": "Country"}, inplace=True)
            df = df.query("Country == 'South Sudan' or Country == 'Ethiopia'")
            df = df[["Country", "Item", "Element", "Year", "Unit", "Value"]]
            df = df[~df.Year.str.contains('-')]
            df["Variable"] = df["Element"] + ", " + df["Item"]
            del df["Element"]
            del df["Item"]
            df.dropna(subset=["Value"], inplace=True)
            dfs.append(df)

    df = pd.concat(dfs)

    df.to_csv(outputFile, index=False, sep="\t")

if __name__ == "__main__":
    clean_FAOSTAT_data(sys.argv[1])
