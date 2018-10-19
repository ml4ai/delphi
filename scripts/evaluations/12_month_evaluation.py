""" Script for cleaning data for 12 month evaluation. """

import sys
import re
import pandas as pd
from glob import glob
from typing import List
from delphi.utils import cd
from delphi.paths import data_dir


def process_climis_crop_production_data(data_dir: str):
    """ Process CliMIS crop production data """

    climis_crop_production_csvs = glob(
        "/".join(
            [
                data_dir,
                "Climis South Sudan Crop Production Data",
                "Crops_EstimatedProductionConsumptionBalance*.csv",
            ]
        )
    )
    state_county_df = pd.read_csv(
        "/".join([data_dir, "ipc_data.csv"]), skipinitialspace=True
    )

    combined_records = []

    for f in climis_crop_production_csvs:
        year = int(f.split("/")[-1].split("_")[2].split(".")[0])
        df = pd.read_csv(f).dropna()
        for i, r in df.iterrows():
            record = {
                "Year": year,
                "Month": None,
                "Source": "CliMIS",
                "Country": "South Sudan",
            }
            region = r["State/County"].strip()

            if region in state_county_df["State"].values:
                record["State"] = region
                record["County"] = None
            else:
                potential_states = state_county_df.loc[
                    state_county_df["County"] == region
                ]["State"]
                if len(potential_states) != 0:
                    record["State"] = potential_states.iloc[0]
                else:
                    record["State"] = None
                record["County"] = region


            for field in r.index:
                if field != "State/County":
                    if "Net Cereal production" in field:
                        record["Variable"] = "Net Cereal Production"
                        record["Value"] = r[field]
                    if field.split()[-1].startswith("("):
                        record["Unit"] = field.split()[-1][1:-1].lower()
                    else:
                        record["Unit"] = None

            combined_records.append(record)

    df = pd.DataFrame(combined_records)
    return df


def process_fao_livestock_data(
    data_dir: str, columns: List[str]
) -> pd.DataFrame:
    csvfile = "/".join(
        [
            "FAO Crop_Livestock Production Data",
            "FAOSTAT_South_Sudan_livestock_data_2014-2016.csv",
        ]
    )

    fao_livestock_csv = "/".join([data_dir, csvfile])

    df = pd.read_csv(
        fao_livestock_csv, usecols=["Element", "Item", "Year", "Unit", "Value"]
    )

    df["Animal"] = df["Item"].str.split(",").str.get(-1)
    df["Product"] = df["Item"].str.split(",").str.get(0)
    df["Variable"] = df["Animal"] + " " + df["Product"] + " " + df["Element"]
    df["Variable"] = df["Variable"].str.lower()
    df["Unit"] = df["Unit"].str.lower()
    df["Source"] = "FAO"
    df["State"] = None
    df["County"] = None
    df["Country"] = "South Sudan"
    df["Month"] = None
    fao_livestock_df = df[columns]
    return fao_livestock_df


def process_fewsnet_data(data_dir, columns: List[str]) -> pd.DataFrame:
    """ Process IPC food security classifications by county for South Sudan. """
    df = pd.read_csv("/".join([data_dir, "ipc_data.csv"]))
    df["Unit"] = "IPC Phase"
    df["Source"] = "FEWSNET"
    df["Variable"] = "IPC Phase Classification"
    df["Country"] = "South Sudan"
    df.rename(str.strip, axis="columns", inplace=True)
    df.rename(columns={"IPC Phase": "Value"}, inplace=True)
    df = df[columns]
    return df


def process_climis_livestock_data(data_dir: str):
    """ Process CliMIS livestock data. """
    climis_livestock_dir = "/".join([str(data_dir), "Climis South Sudan Livestock Data"])
    records = []
    with cd(climis_livestock_dir):
        dirs = glob("*")
        for dir in dirs:
            with cd(dir):
                for filename in glob("*2017.csv"):
                    record = {
                        'Year': 2017,
                        'Variable': "Percentage of householding at least milking one of their livestocks",
                        'County': None,
                        'Country': "South Sudan",
                    }
                    record['State'] = filename.split('_')
                    print(record['State'])
                    string = re.findall('[A-Z][^A-Z]*', record['State'])

def create_combined_table(data_dir: str, columns: List[str]) -> pd.DataFrame:
    climis_crop_production_df = process_climis_crop_production_data(data_dir)
    fao_livestock_df = process_fao_livestock_data(data_dir, columns)
    ipc_df = process_fewsnet_data(data_dir, columns)

    df = pd.concat(
        [climis_crop_production_df, fao_livestock_df, ipc_df], sort=True
    )

    return df[columns]


if __name__ == "__main__":
    columns = [
        "Variable",
        "Year",
        "Value",
        "Unit",
        "Source",
        "State",
        "County",
        "Country",
    ]

    data_dir = str(data_dir / "evaluations" / "12_month")
    process_climis_livestock_data(data_dir)
    combined_table = create_combined_table(data_dir, columns)
    combined_table.to_csv("combined_table.csv", sep="|", index=False)
