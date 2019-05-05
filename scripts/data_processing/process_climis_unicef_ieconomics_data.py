""" Script for cleaning data for 12 month evaluation. """

import os
import re
import sys
import numpy as np
import pandas as pd
from pprint import pprint
from glob import glob
from typing import List, Dict
from delphi.utils.shell import cd
from delphi.paths import data_dir, south_sudan_data
from delphi.utils.fp import grouper
from functools import partial
from itertools import groupby

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

def process_climis_crop_production_data(data_dir: str):
    """ Process CliMIS crop production data """

    climis_crop_production_csvs = glob(
        "{data_dir}/Climis South Sudan Crop Production Data/"
        "Crops_EstimatedProductionConsumptionBalance*.csv"
    )
    state_county_df = pd.read_csv(
        f"{data_dir}/ipc_data.csv", skipinitialspace=True
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

            if region.lower() in state_county_df["State"].str.lower().values:
                record["State"] = region
                record["County"] = None
            else:
                potential_states = state_county_df.loc[
                    state_county_df["County"] == region
                ]["State"]
                record["State"] = (
                    potential_states.iloc[0]
                    if len(potential_states) != 0
                    else None
                )
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


def process_climis_livestock_data(data_dir: str):
    """ Process CliMIS livestock data. """

    records = []

    livestock_data_dir = f"{data_dir}/Climis South Sudan Livestock Data"

    for filename in glob(
        f"{livestock_data_dir}/Livestock Body Condition/*2017.csv"
    ):
        records += process_file_with_single_table(
            filename,
            lambda ind: f"Percentage of {filename.split('_')[-3].lower()} with body condition {ind.lower()}",
            lambda f: f.split("_")[-2],
        )


    for filename in glob(
        f"{livestock_data_dir}/Livestock Production/*2017.csv"
    ):
        records += process_file_with_single_table(
            filename,
            lambda ind: "Percentage of householding at least milking one of their livestocks",
            lambda f: f.split("_")[1],
        )

    disease_acronym_dict = {
        "FMD": "Foot and Mouth Disease (FMD)",
        "LSD": "Lumpy Skin Disease (LSD)",
        "CBPP": "Contagious Bovine Pleuropneumonia (CBPP)",
        "CCPP": "Contagious Caprine Pleuropneumonia (CCPP)",
        "NC": "NC",
        "PPR": "Peste des Petits Ruminants (PPR)",
        "Others": "Other diseases",
    }

    func = (
        lambda k, i: f"Percentage of livestock with {disease_acronym_dict[k]} that are {i.lower().strip()}"
    )
    livestock_disease_header_dict = {
        k: partial(func, k) for k in disease_acronym_dict
    }

    livestock_migration_header_dict = {
        "Livestock migration": lambda i: f"Percentage of livestock migrating {i.split()[-1].lower()}",
        "Distance covered": lambda i: "Distance covered by migrating livestock",
        "Proportion of livestock that migrated": lambda i: "Percentage of livestock that migrated",
        "Migration normal at this time of the year": lambda i: f"Migration normal at this time of year, {i}",
        "Duration in months when the migrated animals are expected to be back after": lambda i: "Duration in months when the migrated animals are expected to be back after",
        "Reasons for livestock migration": lambda i: f"Percentage of livestock migrating due to {i.lower()}",
    }

    def process_directory(dirname, header_dict):
        return pd.concat(
            [
                df
                for df in [
                    process_file_with_multiple_tables(f, header_dict)
                    for f in glob(f"{livestock_data_dir}/{dirname}/*2017.csv")
                ]
                if df is not None
            ]
        )

    func2 = (
        lambda k, i: f"{k.replace('animals', i.lower()).replace('stock', 'stock of '+i.lower()).replace('animal', i.lower())}"
    )
    livestock_ownership_headers = [
        "Average current stock per household",
        "Average number of animals born per household during last 4 weeks",
        "Average number of animals acquired per household during last 4 weeks (dowry, purchase, gift)",
        "Average number of animals given out as bride price/gift per household during last 4 weeks per household",
        "Average number of animals sold per household during last 4 weeks household",
        "Average price of animal sold (SSP)",
        "Average number of animals exchanged for grain per household during last 4 weeks",
        "Average number of animals died/slaughtered/lost per  household during last 4 weeks",
    ]

    livestock_ownership_header_dict = {
        k: partial(func2, k) for k in livestock_ownership_headers
    }
    ownership_df = process_directory(
        "Livestock Ownership", livestock_ownership_header_dict
    )

    disease_df = process_directory(
        "Livestock Diseases", livestock_disease_header_dict
    )

    livestock_migration_df = process_directory(
        "Livestock Migration", livestock_migration_header_dict
    )

    livestock_pasture_header_dict = {
        "Pasture condtion": lambda i: f"Percentage of livestock pasture in {i.lower()} condition",
        "Pasture condition compared to similar time in a normal year": lambda i: f"Percentage of livestock pasture in {i.lower()} condition compared to a similar time in a normal year",
        "Browse condition": lambda i: f"Percentage of livestock pasture in {i.lower()} browse condition",
        "Browse condition compared to similar time in a normal year": lambda i: f"Percentage of livestock pasture in {i.lower()} browse condition compared to a similar time in a normal year",
        "Presence of constraints in accessing forage": lambda i: f"Percentage reporting the {('presence' if i=='Yes' else 'absence')} of constraints in accessing forage",
        "Main forage constraints": lambda i: f"Percentage reporting {i.lower()} as the main forage constraint",
    }
    livestock_pasture_df = process_directory(
        "Livestock Pasture", livestock_pasture_header_dict
    )
    livestock_water_sources_header_dict = {
        "Main water sources": lambda i: f"Percentage of livestock whose main water source is {i.lower()}",
        "Number of days livestock have been watered in the last 7 days": lambda i: f"Number of days {i.lower()} have been watered in the last 7 days",
    }

    livestock_water_sources_df = process_directory(
        "Livestock Water Sources", livestock_water_sources_header_dict
    )

    for filename in glob(f"{livestock_data_dir}/Livestock Loss/*2017.csv"):
        records += process_file_with_single_table(
            filename,
            lambda ind: f"Percentage of {filename.split('_')[-3].lower()} loss accounted for by {ind.lower()}",
            lambda f: f.split("_")[-2],
        )


    for record in records:
        if isinstance(record["Value"], str):
            record["Value"] = record["Value"].replace("%", "")

    livestock_prices_df = pd.concat(
        [
            make_livestock_prices_table(f)
            for f in glob(
                f"{livestock_data_dir}/Livestock Market Prices/*2017.csv"
            )
        ]
    )

    climis_livestock_data_df = pd.concat(
        [
            pd.DataFrame(records),
            disease_df,
            ownership_df,
            livestock_prices_df,
            livestock_migration_df,
            livestock_pasture_df,
            livestock_water_sources_df,
        ]
    )
    return climis_livestock_data_df


def process_climis_import_data(data_dir: str) -> pd.DataFrame:
    dfs = []
    for f in glob(f"{data_dir}/CliMIS Import Data/*.csv"):
        df = pd.read_csv(f, names=range(1, 13), header=0, thousands=",")
        df = df.stack().reset_index(name="Value")
        df.columns = ["Year", "Month", "Value"]
        df["Month"] = df["Month"].astype(int)
        df["Year"] = df["Year"].astype(int)
        dfs.append(df)
    df = (
        pd.concat(dfs)
        .pivot_table(values="Value", index=["Year", "Month"], aggfunc=np.sum)
        .reset_index()
    )

    df.columns = ["Year", "Month", "Value"]
    df["Variable"] = "Total amount of cereal grains imported"
    df["Unit"] = "metric tonne"
    df["Country"] = "South Sudan"
    df["County"] = None
    df["State"] = None
    return df


def process_climis_rainfall_data(data_dir: str) -> pd.DataFrame:
    dfs = []
    # Read CSV files first
    for f in glob(f"{data_dir}/CliMIS South Sudan Rainfall Data in"
                  " Millimeters/*.csv"):
        # Get the name of the table without path and extension
        table_name = os.path.basename(f)[:-4]
        # Get state and year from groups
        pattern = r'^(.*) ([0-9]+) Rainfall'
        state, year = re.match(pattern, table_name).groups()
        df = pd.read_csv(f, header=0, thousands=",")
        cols = ['Variable', 'Year', 'Month', 'Value', 'Unit', 'Source',
                'State', 'County', 'Country']
        df_new = pd.DataFrame(columns=cols)
        df_new['Month'] = range(1, 13)
        df_new['Year'] = int(year)
        df_new['Value'] = df['monthly rainfall data ']
        df_new['Variable'] = 'Rainfall'
        df_new['Unit'] = 'millimeters'
        df_new['County'] = None
        df_new['State'] = state
        df_new['Source'] = 'CliMIS'
        df_new['Country'] = 'South Sudan'
        dfs.append(df_new)
    df1 = pd.concat(dfs)

    # Read XLSX file next
    fname = f'{data_dir}/CliMIS South Sudan Rainfall Data in Millimeters/' + \
        'Rainfall-Early_Warning_6month_Summary-2017-data_table.xlsx'
    df = pd.read_excel(fname, sheet_name='Rainfall Data', header=1)
    cols = ['Variable', 'Year', 'Month', 'Value', 'Unit', 'Source',
            'State', 'County', 'Country']
    df_new = pd.DataFrame(columns=cols)
    states = []
    counties = []
    years = []
    months = []
    values = []
    for row in df.itertuples():
        state, county, year = row[1:4]
        for month in range(1,13):
            value = row[3 + month]
            if pd.isnull(value):
                continue
            states.append(state)
            counties.append(county)
            years.append(year)
            months.append(month)
            values.append(value)
    df_new['Year'] = years
    df_new['Month'] = months
    df_new['Value'] = values
    df_new['County'] = counties
    df_new['State'] = states
    df_new['Variable'] = 'Rainfall'
    df_new['Unit'] = 'millimeters'
    df_new['Source'] = 'CliMIS'
    df_new['Country'] = 'South Sudan'

    df = pd.concat([df1, df_new])
    return df

def process_UNHCR_data(data_dir: str):
    df = pd.read_table(f"{data_dir}/UNHCR Refugee Data/RefugeeData.tsv",
            index_col=0,
            parse_dates=True, infer_datetime_format=True)
    df["Year"] = df.index.year
    df["Month"] = df.index.month
    df.rename(columns = {"individuals":"Value"}, inplace=True)
    df["Country"] = "South Sudan"
    df["State"] = None
    df["County"] = None
    df["Source"] = "UNHCR"
    df["Unit"] = None
    df["Variable"] = "Number of refugees"
    del df["unix_timestamp"]
    return df


def create_combined_table(data_dir: str, columns: List[str]) -> pd.DataFrame:
    climis_crop_production_df = process_climis_crop_production_data(data_dir)
    climis_livestock_data_df = process_climis_livestock_data(data_dir)
    climis_import_data_df = process_climis_import_data(data_dir)
    climis_rainfall_data_df = process_climis_rainfall_data(data_dir)
    UNHCR_data_df = process_UNHCR_data(data_dir)

    # Severe acute malnutrition and inflation rate indicators from PDFs
    pdf_indicators_df = pd.read_table(f"{data_dir}/indicator_data_from_pdfs.tsv")

    df = pd.concat(
        [
            climis_crop_production_df,
            climis_livestock_data_df,
            climis_import_data_df,
            climis_rainfall_data_df,
            pdf_indicators_df,
            UNHCR_data_df,
        ],
        sort=True,
    )

    return df[columns]


if __name__ == "__main__":
    columns = [
        "Variable",
        "Year",
        "Month",
        "Value",
        "Unit",
        "Source",
        "State",
        "County",
        "Country",
    ]

    data_dir = str(data_dir / "evaluations" / "12_month")
    df = create_combined_table(data_dir, columns)
    df["Year"] = df["Year"].astype(int)
    df = df[(df.Year < 2017) | ((df.Year == 2017) & (df.Month <= 4))]
    df.to_csv("data/south_sudan_data_climis_unicef_ieconomics.tsv", index=False, sep="\t")
