import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine

ENGINE = create_engine(f"sqlite:///{sys.argv[5]}", echo=False)

def insert_table(df, table_name):
    df.to_sql(table_name, con=ENGINE, if_exists="replace")


def create_indicator_table(indicator_table):
    df = pd.read_table(indicator_table, index_col=False)
    insert_table(df, "indicator")


def create_adjectiveData_table(adjectiveData):
    df = pd.read_csv(adjectiveData, index_col=False, delim_whitespace=True)
    insert_table(df, "gradableAdjectiveData")


def create_concept_to_indicator_mapping_table(mapping_table):
    df = pd.read_table(
        mapping_table,
        usecols=[1, 2, 3, 4],
        names=["Concept", "Source", "Indicator", "Score"],
        dtype={
            "Concept": str,
            "Source": str,
            "Indicator": str,
            "Score": np.float64,
        },
    )
    df.Indicator = df.Indicator.str.replace("\\\/","/")
    df = df[df['Source'] == 'mitre12']
    df.Indicator = df.Indicator.str.replace("MITRE12/","")

    insert_table(df, "concept_to_indicator_mapping")

def create_dssat_data_table(dssat_aggregated_data_dir):
    state_dict = {"NBG":"Northern Bahr El Ghazal","Unity":"Unity"}
    crop_dict = {"MAIZ":"maize", "SORG":"sorghum"}
    dfs = []
    for filename in ("NBG_MAIZ", "NBG_SORG", "Unity_MAIZ", "Unity_SORG"):
        df = pd.read_csv(f"{dssat_aggregated_data_dir}/{filename}.csv", usecols=[0, 1, 2])
        state, crop = filename.split("_")
        df["State"], df["Crop"]  = state_dict[state], crop_dict[crop]
        df["Source"] = "DSSAT"
        dfs.append(df)

    insert_table(pd.concat(dfs), "dssat")

if __name__ == "__main__":
    create_indicator_table(sys.argv[1])
    create_concept_to_indicator_mapping_table(sys.argv[2])
    create_adjectiveData_table(sys.argv[3])
    create_dssat_data_table(sys.argv[4])
