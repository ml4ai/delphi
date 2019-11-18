import sys
from pathlib import Path

import numpy as np
import pandas as pd

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
    df.loc[0, "Destination state"] = "Eastern Lakes"
    df.loc[0, "Destination county"] = "Awerial South"

    df.loc[1, "Source state"] = "Yei River"
    df.loc[1, "Source county"] = "Yei"
    df.loc[1, "Destination country"] = "South Sudan"
    df.loc[1, "Destination state"] = "Jonglei"
    df.loc[1, "Destination county"] = "Bor"
    df.loc[
        df["Variable"] == "Incoming Migrants", "Source country"
    ] = "Ethiopia"
    df.loc[
        df["Variable"] == "Incoming Migrants", "Destination country"
    ] = "South Sudan"
    df["Source"] = "Migration Curation Experiment"
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

    return df


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
    df.drop([3, 5, 10, 21, 27], inplace=True)

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
    df["Source"] = "Migration Curation Experiment"
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

    return df


def clean_62801_data():
    df = pd.read_csv(
        "data/raw/migration/Initial annotation exercise for migration use case - 62801 - dep var.tsv",
        sep="\t",
    )
    df.loc[22, "Value count"] = 700
    df = df[~np.isnan(df["Value count"].astype(float))]
    df.drop(
        df.columns[[0, 1, 2, 4, 5, 8, 9, 12, 13, 16, 19, 20]],
        axis=1,
        inplace=True,
    )
    df["Value count"] = df["Value count"].astype(float)

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

    df["Unit"] = "people"

    df["Start year"].fillna(value=-1, inplace=True, downcast="infer")
    df["Start month"].fillna(value=0, inplace=True, downcast="infer")
    df["End year"].fillna(value=-1, inplace=True, downcast="infer")
    df["End month"].fillna(value=0, inplace=True, downcast="infer")

    df.loc[4, "Value count"] = df.loc[4, "Value count"] * 30
    df.loc[7:9, "Value count"] = (df.loc[7:9, "Value count"] * 365653.0) / 100

    df.loc[7:9, "End year"] = 2017
    df.loc[7:9, "End month"] = 3

    df.loc[16:19, "Start year"] = 2016
    df.loc[16:19, "Start month"] = 9

    df["Variable"] = "Outgoing Migrants"
    df.drop([3, 11], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["Source country"] = "South Sudan"
    df["Source county"] = "None"
    df["Source state"] = "None"
    df["Destination country"] = "Ethiopia"
    df["Destination county"] = "None"
    df["Destination state"] = "Gambella"

    df.loc[5, "Source state"] = "Upper Nile"
    df.loc[6, "Source state"] = "Jonglei"
    df.loc[7, "Source state"] = "Unity"
    df.loc[15, "Source state"] = "Boma"
    df.loc[15, "Source county"] = "Pochala"
    df.loc[[3, 5, 6, 7], "Destination state"] = "None"

    df["Source"] = "Migration Curation Experiment"
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

    return df


def clean_62803_data():
    df = pd.read_csv(
        "data/raw/migration/Initial annotation exercise for migration use case - 62803 - dep var.tsv",
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

    df["Unit"] = "people"

    df["Start year"].fillna(value=-1, inplace=True, downcast="infer")
    df["Start month"].fillna(value=0, inplace=True, downcast="infer")
    df["End year"].fillna(value=-1, inplace=True, downcast="infer")
    df["End month"].fillna(value=0, inplace=True, downcast="infer")

    df.loc[3, "Value count"] = df.loc[3, "Value count"] * 30
    df.loc[4:6, "Value count"] = (df.loc[4:6, "Value count"] * 361991.0) / 100

    df.loc[4:6, "End year"] = 2017
    df.loc[4:6, "End month"] = 4

    df["Variable"] = "Outgoing Migrants"
    df.drop([7, 9], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["Source country"] = "South Sudan"
    df["Source county"] = "None"
    df["Source state"] = "None"
    df["Destination country"] = "Ethiopia"
    df["Destination county"] = "None"
    df["Destination state"] = "Gambella"

    df.loc[4, "Source state"] = "Upper Nile"
    df.loc[5, "Source state"] = "Jonglei"
    df.loc[6, "Source state"] = "Unity"
    df.loc[8, "Source state"] = "Boma"
    df.loc[8, "Source county"] = "Pochala"
    df.loc[3:6, "Destination state"] = "None"

    df["Source"] = "Migration Curation Experiment"
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

    return df


def clean_63604_data():
    df = pd.read_csv(
        "data/raw/migration/Initial annotation exercise for migration use case - 63604 - dep var.tsv",
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

    df["Unit"] = "people"

    df["Start year"].fillna(value=-1, inplace=True, downcast="infer")
    df["Start month"].fillna(value=0, inplace=True, downcast="infer")
    df["End year"].fillna(value=-1, inplace=True, downcast="infer")
    df["End month"].fillna(value=0, inplace=True, downcast="infer")

    df["Variable"] = "Outgoing Migrants"
    df.drop([1, 5, 6, 7], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["Source country"] = "South Sudan"
    df["Source county"] = "None"
    df["Source state"] = "None"
    df["Destination country"] = "Ethiopia"
    df["Destination county"] = "None"
    df["Destination state"] = "None"

    df.loc[2, "Destination state"] = "Beneshangul Gumuz"
    df.loc[3, "Destination state"] = "Gambella"

    df["Source"] = "Migration Curation Experiment"
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

    return df


def clean_UNHCR_data():
    df = pd.read_csv(
        "data/raw/migration/Initial annotation exercise for migration use case - UNHCR - dep var.tsv",
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

    df["Unit"] = "people"

    df["Start year"].fillna(value=-1, inplace=True, downcast="infer")
    df["Start month"].fillna(value=0, inplace=True, downcast="infer")
    df["End year"].fillna(value=-1, inplace=True, downcast="infer")
    df["End month"].fillna(value=0, inplace=True, downcast="infer")

    df["Variable"] = "Outgoing Migrants"
    df.loc[5, "Value count"] = df.loc[5, "Value count"] * 31
    df.loc[6:8, "Value count"] = (df.loc[6:8, "Value count"] * 361000.0) / 100

    df.loc[6:8, "End year"] = 2017
    df.loc[6:8, "End month"] = 3

    df.loc[17:20, "Start year"] = 2016
    df.loc[17:20, "Start month"] = 9

    df.drop([0], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["Source country"] = "South Sudan"
    df["Source county"] = "None"
    df["Source state"] = "None"
    df["Destination country"] = "Ethiopia"
    df["Destination county"] = "None"
    df["Destination state"] = "None"

    df.loc[4, "Source state"] = "Upper Nile"
    df.loc[5, "Source state"] = "Jonglei"
    df.loc[6, "Source state"] = "Unity"
    df.loc[0:6, "Destination state"] = "Gambella"
    df.loc[9:14, "Destination state"] = "Gambella"

    df["Source"] = "Migration Curation Experiment"
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

    return df


if __name__ == "__main__":
    combined_df = pd.concat([
        clean_reachjongleijan_data(),
        clean_54660_data(),
        clean_62801_data(),
        clean_62803_data(),
        clean_63604_data(),
        clean_UNHCR_data(),
    ])
    combined_df.to_csv(sys.argv[1], sep="\t", index=False)

