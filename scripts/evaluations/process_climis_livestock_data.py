import re
import pandas as pd
import numpy as np
from itertools import groupby
from delphi.utils.fp import grouper
from typing import Dict


def get_state_from_filename(filename, get_state_func):
    return " ".join(re.findall("[A-Z][^A-Z]*", get_state_func(filename)))


def process_file_with_single_table(
    filename, variable_name_func, get_state_func, country="South Sudan"
):
    records = []
    df = pd.read_csv(
        filename, index_col=0, names=range(12), header=0, skipinitialspace=True
    )
    for ind in df.index:
        for column in df.columns:
            record = {
                "Variable": variable_name_func(ind),
                "Month": column + 1,
                "Value": df.loc[ind][column],
                "State": get_state_from_filename(filename, get_state_func),
                "Country": country,
            }
            set_defaults(record)
            records.append(record)
    return records


def set_climis_south_sudan_default_params(
    filename, df, get_state_func=lambda x: x.split("_")[-2]
):
    df["Country"] = "South Sudan"
    df["Source"] = "CLiMIS"
    df["Year"] = int(filename.split(".")[0].split("_")[-1])
    df["State"] = get_state_from_filename(filename, get_state_func)
    return df


def make_livestock_prices_table(filename):
    df = pd.read_csv(
        filename,
        index_col=[0, 1],
        header=0,
        names=["County", "Market"] + list(range(1, 13)),
        skipinitialspace=True,
        thousands=",",
    )
    df = df.stack().reset_index(name="Value")
    df.columns = ["County", "Market", "Month", "Value"]
    df = df.pivot_table(values="Value", index=["County", "Month"])
    df = set_climis_south_sudan_default_params(filename, df)
    df["Unit"] = "SSP"
    df["Variable"] = f"Average price of {filename.split('_')[-3].lower()}"
    df = df.reset_index()
    return df


def set_defaults(record: Dict):
    record.update(
        {
            "Year": 2017,
            "Country": "South Sudan",
            "Unit": "%",
            "Source": "CliMIS",
            "County": None,
        }
    )


def make_group_dict(groups):
    return {k[0][0]: g for k, g in grouper(groups, 2)}


def make_df_from_group(k, v, index_func):
    df = pd.DataFrame(v)
    df.set_index(0, inplace=True)
    df.index = [index_func(k, i) for i in df.index]
    df = df.stack().reset_index(name="Value")
    df.columns = ["Variable", "Month", "Value"]
    df["Month"] = df["Month"].astype(int)
    return df


def process_file_with_multiple_tables(filename, header_dict):
    dfs = []
    df = pd.read_csv(filename, index_col=0, names=range(12), header=0)

    # Define a grouping key function to split the CSV by the header rows
    grouping_key_function = lambda _tuple: _tuple[1][1:].isna().all()
    iterrows = filter(lambda r: r[1][0] != "", df.iterrows())
    key_group_tuples = groupby(iterrows, grouping_key_function)
    groups = [
        [
            [x[0].strip()] + x[1].values.tolist()
            for x in list(g)
            if isinstance(x[0], str)
        ]
        for k, g in key_group_tuples
    ]

    for k, v in make_group_dict(groups).items():
        if v is not None:
            df = make_df_from_group(
                k, v, lambda k, i: header_dict.get(k.strip(), lambda x: k)(i)
            )
            df["Value"] = df["Value"].replace(" ", np.nan)
            df = df.dropna()
            df["County"] = None
            df = set_climis_south_sudan_default_params(filename, df)

            if len(df.Value.values) > 0 and any(
                map(lambda v: "%" in v, df["Value"].values)
            ):
                df.Value = df.Value.str.replace("%", "")
                df["Unit"] = "%"
            else:
                df["Unit"] = None
            if len(df["Variable"].values) > 0:
                if "SSP" in df["Variable"].values[0]:
                    df["Variable"] = (
                        df["Variable"].str.replace("\(SSP\)", "").str.strip()
                    )
                    df["Unit"] = "SSP"

            if len(df.Value.values) > 0 and "-" in df.Value.values[0]:
                # For percentage ranges, take the mean value
                df.Value = (
                    df.Value.str.strip()
                    .str.split("-")
                    .map(lambda x: list(map(float, x)))
                    .map(lambda x: np.mean(x))
                )

            dfs.append(df)

    if len(dfs) > 0:
        return pd.concat(dfs)
    else:
        return None
