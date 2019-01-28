import sys
from glob import glob
import numpy as np
import pandas as pd
from delphi.utils.fp import take, compose
from itertools import chain


def process_dssat_table(f):
    expansion_dict = {
        "nbg": "Northern Bahr el Ghazal",
        "unity": "Unity",
        "maiz": "Maize",
        "sorg": "Sorghum",
    }

    df = pd.read_csv(
        f, index_col=0, parse_dates=True, infer_datetime_format=True
    )
    del df["Harvested Area (ha)"]
    filename_parts = f.split("/")[-1].split("_")
    df["State"] = expansion_dict[filename_parts[0]]
    df["Variable"] = f"Production of {expansion_dict[filename_parts[1]]}"
    df.rename(columns={"Production (t)": "Value"}, inplace=True)
    df.Value = df.Value.astype(int)
    return df


def process_dssat_data(input, output):
    csvs = glob(input + "/*historical*")
    df = compose(pd.concat, list, chain)(map(process_dssat_table, csvs))
    df = (
        df.groupby(["Variable", "State", pd.Grouper(freq="M")])
        .sum()
        .reset_index(("Variable", "State"))
    )
    df["Country"] = "South Sudan"
    df["Unit"] = "tons"
    df["County"] = None
    df["Source"] = "DSSAT"
    df["Year"] = df.index.year
    df["Month"] = df.index.month
    df.to_csv(output, sep="\t", index=False)


if __name__ == "__main__":
    process_dssat_data(sys.argv[1], sys.argv[2])
