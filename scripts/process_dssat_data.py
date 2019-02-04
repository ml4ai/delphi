import sys
from glob import glob
from itertools import chain
from functools import partial
import numpy as np
import pandas as pd
from delphi.utils.fp import take, compose

EXPANSION_DICT = {
    "nbg": "Northern Bahr el Ghazal",
    "unity": "Unity",
    "maiz": "Maize",
    "sorg": "Sorghum",
}


def process_dssat_table(variable_str, f):
    df = pd.read_csv(
        f, index_col=0, parse_dates=True, infer_datetime_format=True
    )
    if "Harvested Area (ha)" in df.columns:
        del df["Harvested Area (ha)"]
    filename_parts = f.split("/")[-1].split("_")
    method = filename_parts[2]
    df["State"] = EXPANSION_DICT[filename_parts[0]]
    variable = variable_str.split(" (")[0]
    df["Variable"] = f"{method.capitalize()} {variable} ({EXPANSION_DICT[filename_parts[1]]})"
    df.rename(columns={variable_str: "Value"}, inplace=True)
    df.Value = df.Value.astype(int)
    return df


def process_dssat_data(func, directory, unit):
    csvs = glob(directory)
    df = compose(pd.concat, list, chain)(map(func, csvs))
    if sum:
        df = (
            df.groupby(["Variable", "State", pd.Grouper(freq="M")])
            .sum()
            .reset_index(("Variable", "State"))
        )
    df["Country"] = "South Sudan"
    df["Unit"] = unit
    df["County"] = None
    df["Source"] = "DSSAT"
    df["Year"] = df.index.year
    df["Month"] = df.index.month
    return df


def create_combined_dssat_table(input, output):
    df = pd.concat(
        [
            process_dssat_data(
                partial(process_dssat_table, "Production (t)"),
                input + "/*historical*",
                "tons",
            ),
            process_dssat_data(
                partial(
                    process_dssat_table, "Average Total Daily Rainfall (mm)"
                ),
                input + "/weather/*historical*",
                "mm",
            ),
        ]
    )
    df.to_csv(output, sep="\t", index=False)


if __name__ == "__main__":
    create_combined_dssat_table(sys.argv[1], sys.argv[2])
