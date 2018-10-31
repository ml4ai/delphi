""" Script for cleaning data for 12 month evaluation. """

import sys
import re
import pandas as pd
from glob import glob
from typing import List
from delphi.utils.shell import cd
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

    records = []
    climis_livestock_production_dir = "/".join([str(data_dir), "Climis South Sudan Livestock Data", "Livestock Production"])

    with cd(climis_livestock_production_dir):
        # dirs = glob("*")
        # for dir in dirs:
            # with cd(dir):
        # print("climis_livestock_dir:" + climis_livestock_dir)
        for filename in glob('*2017.csv'):
            # print('filename:' + filename)
            df = pd.read_csv(filename, index_col=0)
            # print(df.index)
            for column in df.columns:
                # print(column)
                record = {
                    'Year': 2017,
                    'Variable': "Percentage of householding at least milking one of their livestocks",
                    'County': None,
                    'Country': "South Sudan",
                    'Unit': '%',
                    'Month': column,
                    'Value': df.loc['Households '][column],
                    'Source': 'CliMIS'
                    }
                state_without_spaces = filename.split('_')[1]
                # print(record)
                record['State'] = ' '.join(re.findall('[A-Z][^A-Z]*', state_without_spaces))
                records.append(record)
                
    climis_livestock_bodycondition_dir = "/".join([str(data_dir), "Climis South Sudan Livestock Data", "Livestock Body Condition"])
    with cd(climis_livestock_bodycondition_dir):
        for filename in glob('*2017.csv'):
            df = pd.read_csv(filename, index_col=0)
            for i in df.index:
                for column in df.columns:
                    record = {
                        'Year': 2017,
                        'County': None,
                        'Country': "South Sudan",
                        'Unit': '%',
                        'Month': column,
                        'Value': df.loc[i][column],
                        'Source': 'CliMIS'
                        }
                    state_without_spaces = filename.split('_')[-2]
                    # print(record)
                    record['State'] = ' '.join(re.findall('[A-Z][^A-Z]*', state_without_spaces))     
                    record['Variable'] = "Percentage of " + filename.split('_')[-3] + " in body condiction of " + i
                    records.append(record)


    climis_livestock_diseases_dir = "/".join([str(data_dir), "Climis South Sudan Livestock Data", "Livestock Diseases"])
    with cd(climis_livestock_diseases_dir):
        for filename in glob('*2017.csv'):
            df = pd.read_csv(filename, index_col=0)
            # print(df.loc['FMD'])
            # print()
            # df.rename(str.strip, axis="columns", inplace=True)
            for i, r in df.iterrows():
                # print (i)
                if(i != 'Reported ' and i != 'Vaccinated ' and i != 'Treated '):
                    disease_str = i
                else:
                    for column in df.columns:
                        record = {
                            'Year': 2017,
                            'County': None,
                            'Country': "South Sudan",
                            'Unit': '%',
                            'Month': column,
                            'Value': r[column],
                            'Source': 'CliMIS'
                            }
                        state_without_spaces = filename.split('_')[-2]
                        record['State'] = ' '.join(re.findall('[A-Z][^A-Z]*', state_without_spaces))
                        record['Variable'] = "Percentage of livestocks with disease " + disease_str + " is " + i
                        records.append(record)


    climis_livestock_ownership_dir = "/".join([str(data_dir), "Climis South Sudan Livestock Data", "Livestock Ownership"])
    with cd(climis_livestock_ownership_dir):
        for filename in glob('*2017.csv'):
            df = pd.read_csv(filename, index_col=0)
            for i, r in df.iterrows():
                if(i != 'Cattle' and i != 'Goat' and i != 'Sheep' and i != 'Poultry'):
                    quantity_str = i
                else:
                    for column in df.columns:
                        record = {
                            'Year': 2017,
                            'County': None,
                            'Country': "South Sudan",
                            'Month': column,
                            'Value': r[column],
                            'Source': 'CliMIS'
                            }
                        state_without_spaces = filename.split('_')[-2]
                        record['State'] = ' '.join(re.findall('[A-Z][^A-Z]*', state_without_spaces))
                        record['Variable'] = quantity_str.replace('animal', i).replace('stock', i)
                        if(quantity_str == 'Average price of animal sold (SSP)'):
                            record["Unit"] = '$'
                        else:
                            record["Unit"] = None

                        records.append(record)


    climis_livestock_loss_dir = "/".join([str(data_dir), "Climis South Sudan Livestock Data", "Livestock Loss"])
    with cd(climis_livestock_loss_dir):
        for filename in glob('*2017.csv'):
            df = pd.read_csv(filename, index_col=0)
            for i in df.index:
                for column in df.columns:
                    record = {
                        'Year': 2017,
                        'County': None,
                        'Country': "South Sudan",
                        'Unit': '%',
                        'Month': column,
                        'Value': df.loc[i][column],
                        'Source': 'CliMIS'
                        }
                    state_without_spaces = filename.split('_')[-2]
                    # print(record)
                    record['State'] = ' '.join(re.findall('[A-Z][^A-Z]*', state_without_spaces))     
                    record['Variable'] = "Percentage of " + filename.split('_')[-3] + " suffer from " + i
                    records.append(record)

    climis_livestock_MarketPrices_dir = "/".join([str(data_dir), "Climis South Sudan Livestock Data", "Livestock Market Prices"])
    with cd(climis_livestock_MarketPrices_dir):
        for filename in glob('*2017.csv'):
            df = pd.read_csv(filename, index_col=0)
            for i, r in df.iterrows():
                for column in df.columns:
                    if(column!= 'Market'):
                        record = {
                            'Year': 2017,
                            'Country': "South Sudan",
                            'Unit': '$',
                            'Month': column,
                            'Value': r[column],
                            'Source': 'CliMIS'
                            }
                        state_without_spaces = filename.split('_')[-2]
                        record['County'] = i
                        record['State'] = ' '.join(re.findall('[A-Z][^A-Z]*', state_without_spaces))     
                        record['Variable'] = "Price of " + filename.split('_')[-3] + " in Market " + r['Market']
                        records.append(record)

    climis_livestock_migration_dir = "/".join([str(data_dir), "Climis South Sudan Livestock Data", "Livestock Migration"])
    with cd(climis_livestock_migration_dir):
        for filename in glob('*2017.csv'):
            df = pd.read_csv(filename, index_col=0)
            row_id = 0
            for i, r in df.iterrows():
    
                if(row_id == df.index.get_loc("Distance covered") + 1):
                    for column in df.columns:
                        record = {
                            'Year': 2017,
                            'Variable': "Livestock migration distance covered",
                            'County': None,
                            'Country': "South Sudan",
                            'Unit': 'mile',
                            'Month': column,
                            'Value': r[column],
                            'Source': 'CliMIS'
                            }
                        state_without_spaces = filename.split('_')[-2]
                        record['State'] = ' '.join(re.findall('[A-Z][^A-Z]*', state_without_spaces))
                        records.append(record)

                if(row_id == df.index.get_loc("Proportion of livestock that migrated") + 1):
                    for column in df.columns:
                        record = {
                            'Year': 2017,
                            'Variable': "Proportion of livestock that migrated",
                            'County': None,
                            'Country': "South Sudan",
                            'Unit': '%',
                            'Month': column,
                            'Value': r[column],
                            'Source': 'CliMIS'
                            }
                        state_without_spaces = filename.split('_')[-2]
                        record['State'] = ' '.join(re.findall('[A-Z][^A-Z]*', state_without_spaces))
                        records.append(record)

                if(row_id == df.index.get_loc(" Migration normal at this time of the year") + 1):
                    for column in df.columns:
                        record = {
                            'Year': 2017,
                            'Variable': "Proportion of livestock that migrated normally at this time of the year",
                            'County': None,
                            'Country': "South Sudan",
                            'Unit': '%',
                            'Month': column,
                            'Value': r[column],
                            'Source': 'CliMIS'
                            }
                        state_without_spaces = filename.split('_')[-2]
                        record['State'] = ' '.join(re.findall('[A-Z][^A-Z]*', state_without_spaces))
                        records.append(record)

                if(row_id == df.index.get_loc(" Migration normal at this time of the year") + 2):
                    for column in df.columns:
                        record = {
                            'Year': 2017,
                            'Variable': "Proportion of livestock that migrated abnormally at this time of the year",
                            'County': None,
                            'Country': "South Sudan",
                            'Unit': '%',
                            'Month': column,
                            'Value': r[column],
                            'Source': 'CliMIS'
                            }
                        state_without_spaces = filename.split('_')[-2]
                        record['State'] = ' '.join(re.findall('[A-Z][^A-Z]*', state_without_spaces))
                        records.append(record)

                if(row_id == df.index.get_loc(" Duration in months when the migrated animals are expected to be back after  ") + 1):
                    for column in df.columns:
                        record = {
                            'Year': 2017,
                            'Variable': "Duration in months when the migrated animals are expected to be back after",
                            'County': None,
                            'Country': "South Sudan",
                            'Unit': "month",
                            'Month': column,
                            'Value': r[column],
                            'Source': 'CliMIS'
                            }
                        state_without_spaces = filename.split('_')[-2]
                        record['State'] = ' '.join(re.findall('[A-Z][^A-Z]*', state_without_spaces))
                        records.append(record)

                if(i == 'Migration In ' or i == 'Migration Out' or i == 'No Migration'):
                    for column in df.columns:
                        record = {
                            'Year': 2017,
                            'County': None,
                            'Country': "South Sudan",
                            'Unit': '%',
                            'Month': column,
                            'Value': r[column],
                            'Source': 'CliMIS'
                            }
                        state_without_spaces = filename.split('_')[-2]
                        record['State'] = ' '.join(re.findall('[A-Z][^A-Z]*', state_without_spaces))
                        record['Variable'] = "Livestocks Percentage of " + i
                        records.append(record)
              

                if(i == 'Pasture' or i == 'Water' or i == 'Conflict / Insecurity' or i == 'Disease' or i == 'Wild Conflict' or i == 'Others'):
                    for column in df.columns:
                        record = {
                            'Year': 2017,
                            'County': None,
                            'Country': "South Sudan",
                            'Unit': '%',
                            'Month': column,
                            'Value': r[column],
                            'Source': 'CliMIS'
                            }
                        state_without_spaces = filename.split('_')[-2]
                        record['State'] = ' '.join(re.findall('[A-Z][^A-Z]*', state_without_spaces))
                        record['Variable'] = "Percentage of livestock migration due to " + i
                        records.append(record)

                row_id += 1


    climis_livestock_data_df = pd.DataFrame(records)
    # print(climis_livestock_data_df)
    return climis_livestock_data_df

def create_combined_table(data_dir: str, columns: List[str]) -> pd.DataFrame:
    climis_crop_production_df = process_climis_crop_production_data(data_dir)
    fao_livestock_df = process_fao_livestock_data(data_dir, columns)
    ipc_df = process_fewsnet_data(data_dir, columns)
    climis_livestock_data_df = process_climis_livestock_data(data_dir)

    df = pd.concat(
        [climis_crop_production_df, fao_livestock_df, ipc_df, climis_livestock_data_df], sort=True
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
    combined_table = create_combined_table(data_dir, columns)
    combined_table.to_csv("combined_table.csv", sep="|", index=False)
