import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sqlalchemy import create_engine

ENGINE = create_engine(f"sqlite:///{sys.argv[5]}", echo=False, pool_pre_ping=True)


def insert_table(df, table_name):
    df.to_sql(table_name, con=ENGINE, if_exists="replace")


def create_indicator_table(indicator_table):
    df = pd.read_csv(indicator_table, index_col=False, sep="\t",
            dtype={"Country": str, "Source": str})
    df["Country"].fillna(value="None", inplace=True, downcast="infer")
    df["County"].fillna(value="None", inplace=True, downcast="infer")
    df["Month"].fillna(value=0, inplace=True, downcast="infer")
    df["Source"].fillna(value="None", inplace=True, downcast="infer")
    df["State"].fillna(value="None", inplace=True, downcast="infer")
    df["Unit"].fillna(value="None", inplace=True, downcast="infer")
    df["Variable"].fillna(value="None", inplace=True, downcast="infer")
    df["Year"].fillna(value=-1, inplace=True, downcast="infer")

    for i,c in enumerate(df['Value']):
        if isinstance(c,str):
            df.loc[i,'Value'] = c.replace(',','')

    df['Country'] = df['Country'].astype(str)
    df['County'] = df['County'].astype(str)
    df['Month'] = df['Month'].astype(int)
    df['Source'] = df['Source'].astype(str)
    df['State'] = df['State'].astype(str)
    df['Unit'] = df['Unit'].astype(str)
    df['Value'] = df['Value'].astype(float)
    df['Variable'] = df['Variable'].astype(str)
    df['Year'] = df['Year'].astype(int)

    insert_table(df, "indicator")


def create_adjectiveData_table(adjectiveData):
    df = pd.read_csv(adjectiveData, index_col=False, delim_whitespace=True)
    insert_table(df, "gradableAdjectiveData")


def create_concept_to_indicator_mapping_table(mapping_table):
    df = pd.read_csv(
        mapping_table,
        usecols=[1, 2, 3, 4],
        names=["Concept", "Source", "Indicator", "Score"],
        dtype={
            "Concept": str,
            "Source": str,
            "Indicator": str,
            "Score": np.float64,
        },
        sep='\t',
    )
    df.Indicator = df.Indicator.str.replace("\\\/", "/")
    df.Indicator = df.Indicator.str.replace("_", " ")
    df = df[df["Source"] == "delphi_db_indicators"]
    df.Indicator = df.Indicator.str.replace(" delphi_db_indicators/", "")

    insert_table(df, "concept_to_indicator_mapping")


def create_dssat_data_table(dssat_aggregated_data_dir):
    state_dict = {"NBG": "Northern Bahr El Ghazal", "Unity": "Unity"}
    crop_dict = {"MAIZ": "maize", "SORG": "sorghum"}
    dfs = []
    for filename in ("NBG_MAIZ", "NBG_SORG", "Unity_MAIZ", "Unity_SORG"):
        df = pd.read_csv(
            f"{dssat_aggregated_data_dir}/{filename}.csv", usecols=[0, 1, 2]
        )
        state, crop = filename.split("_")
        df["State"], df["Crop"] = state_dict[state], crop_dict[crop]
        df["Source"] = "DSSAT"
        dfs.append(df)

    insert_table(pd.concat(dfs), "dssat")


if __name__ == "__main__":
    create_indicator_table(sys.argv[1])
    create_concept_to_indicator_mapping_table(sys.argv[2])
    create_adjectiveData_table(sys.argv[3])
    create_dssat_data_table(sys.argv[4])
