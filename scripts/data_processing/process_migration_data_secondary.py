import numpy as np
import pandas as pd
from pathlib import Path
import sys

data_dir = Path("data")


def clean_data(df):
    df["Country"] = df["Source country"]

    df.loc[df["Variable"] == "Incoming Migrants", "Country"] = df.loc[
        df["Variable"] == "Incoming Migrants", "Destination country"
    ]

    df["State"] = df["Source state"]

    df.loc[df["Variable"] == "Incoming Migrants", "State"] = df.loc[
        df["Variable"] == "Incoming Migrants", "Destination state"
    ]

    df["County"] = df["Source county"]

    df.loc[df["Variable"] == "Incoming Migrants", "County"] = df.loc[
        df["Variable"] == "Incoming Migrants", "Destination county"
    ]

    df["Year"] = df["End year"]
    df["Month"] = df["End month"]

    df.loc[(df["End year"] == -1) & (df["End month"] == 0), "Year"] = df.loc[
        (df["End year"] == -1) & (df["End month"] == 0), "Start year"
    ]

    df.loc[(df["End year"] == -1) & (df["End month"] == 0), "Month"] = df.loc[
        (df["End year"] == -1) & (df["End month"] == 0), "Start month"
    ]

    df.drop(
        df.columns[[0, 1, 2, 3, 4, 5, 10, 11, 12, 13]], axis=1, inplace=True
    )
    df = df.reindex(
        columns=[
            "Country",
            "County",
            "Month",
            "Source",
            "State",
            "Unit",
            "Value",
            "Variable",
            "Year",
        ]
    )

    return df


def secondary_clean(inputFile, outputFile):
    df = pd.read_table(inputFile)
    df = clean_data(df)
    df.to_csv(outputFile, index=False, sep="\t")


if __name__ == "__main__":
    secondary_clean(sys.argv[1], sys.argv[2])
