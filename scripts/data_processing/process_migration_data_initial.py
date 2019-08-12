import numpy as np
import pandas as pd
from pathlib import Path
import sys

data_dir = Path("data")


def clean_reachjongleijan_data():
    df = pd.read_csv(
        "data/raw/migration/Initial annotation exercise for migration use case - ReachJongleiJan - dep var.tsv",
        sep="\t",
    )
    df = df[~np.isnan(df["Value count"])]
    df.drop(
        df.columns[[0, 1, 2, 4, 5, 8, 9, 12, 13, 16, 19, 20]],
        axis=1,
        inplace=True,
    )

    d = {
        "January": 1.0,
        "February": 2.0,
        "March": 3.0,
        "April": 4.0,
        "May": 5.0,
        "June": 6.0,
        "July": 7.0,
        "August": 8.0,
        "September": 9.0,
        "October": 10.0,
        "November": 11.0,
        "December": 12.0,
    }

    df.replace(d, inplace=True)

    df["Start year"].fillna(value=-1, inplace=True, downcast="infer")
    df["Start month"].fillna(value=0, inplace=True, downcast="infer")
    df["End year"].fillna(value=-1, inplace=True, downcast="infer")
    df["End month"].fillna(value=0, inplace=True, downcast="infer")

    c = {
        1: 31,
        2: 28,
        3: 31,
        4: 30,
        5: 31,
        6: 30,
        7: 31,
        8: 31,
        9: 30,
        10: 31,
        11: 30,
        12: 31,
    }

    for i in range(1, 13):
        df.loc[
            (df["Value unit (Amount, Rate, Percentage)"] == "Daily")
            & (df["End month"] == i),
            "Value count",
        ] = (
            df.loc[
                (df["Value unit (Amount, Rate, Percentage)"] == "Daily")
                & (df["End month"] == i),
                "Value count",
            ]
            * c[i]
        )

    df["Unit"] = "people"
    df.reset_index(drop=True, inplace=True)

    df["Variable"] = df["Event trigger text"]

    df.loc[0:1, "Variable"] = "Internally Displaced People"

    df.loc[
        df["Event trigger text"] == "leaving", "Variable"
    ] = "Outgoing Migrants"
    df.loc[
        df["Event trigger text"] == "returning", "Variable"
    ] = "Incoming Migrants"

    df["Source country"] = "South Sudan"
    df["Source county"] = "None"
    df["Source state"] = "None"
    df["Destination country"] = "Ethiopia"
    df["Destination county"] = "None"
    df["Destination state"] = "None"

    df.loc[0, "Source state"] = "Jonglei"
    df.loc[0, "Destination country"] = "South Sudan"
    df.loc[0, "Destination State"] = "Eastern Lakes"
    df.loc[0, "Destination county"] = "Awerial South"

    df.loc[1, "Source state"] = "Yei River"
    df.loc[1, "Source county"] = "Yei"
    df.loc[1, "Destination country"] = "South Sudan"
    df.loc[1, "Destination State"] = "Jonglei"
    df.loc[1, "Destination county"] = "Bor"
    df.loc[
        df["Variable"] == "Incoming Migrants", "Source country"
    ] = "Ethiopia"
    df.loc[
        df["Variable"] == "Incoming Migrants", "Destination country"
    ] = "South Sudan"
    df["Source"] = "ReachJongleiJan"
    df.drop(df.columns[[0, 1, 2, 4]], axis=1, inplace=True)
    df.rename(columns={"Value count": "Value"}, inplace=True)
    df = df.reindex(
        columns=[
            "Source country",
            "Source state",
            "Source county",
            "Destination country",
            "Destination state",
            "Destination county",
            "Source",
            "Unit",
            "Value",
            "Variable",
            "Start year",
            "Start month",
            "End year",
            "End month",
        ]
    )

    df.to_csv(
        str(data_dir / "south_sudan_ReachJongleiJan_migration_data_new.tsv"),
        index=False,
        sep="\t",
    )


def clean_54660_data():
    df = pd.read_csv(
        "data/raw/migration/Initial annotation exercise for migration use case - 54660 - dep var.tsv",
        sep="\t",
    )
    df = df[~np.isnan(df["Value count"])]
    df.drop(
        df.columns[[0, 1, 2, 4, 5, 8, 9, 12, 13, 16, 19, 20]],
        axis=1,
        inplace=True,
    )

    d = {
        "January": 1.0,
        "February": 2.0,
        "March": 3.0,
        "April": 4.0,
        "May": 5.0,
        "June": 6.0,
        "July": 7.0,
        "August": 8.0,
        "September": 9.0,
        "October": 10.0,
        "November": 11.0,
        "December": 12.0,
    }

    df.replace(d, inplace=True)

    df["Start year"].fillna(value=-1, inplace=True, downcast="infer")
    df["Start month"].fillna(value=0, inplace=True, downcast="infer")
    df["End year"].fillna(value=-1, inplace=True, downcast="infer")
    df["End month"].fillna(value=0, inplace=True, downcast="infer")

    c = {
        0: 30,
        1: 31,
        2: 28,
        3: 31,
        4: 30,
        5: 31,
        6: 30,
        7: 31,
        8: 31,
        9: 30,
        10: 31,
        11: 30,
        12: 31,
    }

    for i in range(13):
        df.loc[
            (df["Value unit (Amount, Rate, Percentage)"] == "Daily")
            & (df["End month"] == i),
            "Value count",
        ] = (
            df.loc[
                (df["Value unit (Amount, Rate, Percentage)"] == "Daily")
                & (df["End month"] == i),
                "Value count",
            ]
            * c[i]
        )

    df.loc[5, "Value count"] = df.loc[5, "Value count"] * 31
    df["Unit"] = "people"
    df.drop([3, 5, 10, 14, 21, 27], inplace=True)

    df.loc[7, "Start year"] = 2017
    df.loc[7, "Start month"] = 3
    df.loc[7, "End year"] = 2017
    df.loc[7, "End month"] = 3

    df.loc[15:18, "Start year"] = 2016
    df.loc[15:18, "Start month"] = 9

    df.reset_index(drop=True, inplace=True)

    df["Variable"] = "Outgoing Migrants"

    df["Source country"] = "South Sudan"
    df["Source county"] = "None"
    df["Source state"] = "None"
    df["Destination country"] = "Ethiopia"
    df["Destination county"] = "None"
    df["Destination state"] = "Gambella"

    df.loc[[5, 6, 12], "Destination state"] = "None"
    df["Source"] = "54660"
    df.drop(df.columns[[0, 1, 2, 4]], axis=1, inplace=True)

    df.rename(columns={"Value count": "Value"}, inplace=True)
    df = df.reindex(
        columns=[
            "Source country",
            "Source state",
            "Source county",
            "Destination country",
            "Destination state",
            "Destination county",
            "Source",
            "Unit",
            "Value",
            "Variable",
            "Start year",
            "Start month",
            "End year",
            "End month",
        ]
    )

    df.to_csv(
        str(data_dir / "south_sudan_54660_migration_data_new.tsv"),
        index=False,
        sep="\t",
    )


if __name__ == "__main__":
    clean_reachjongleijan_data()
    clean_54660_data()
