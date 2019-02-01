import numpy as np
import pandas as pd
from delphi.paths import (
    data_dir,
    db_path,
    south_sudan_data,
    adjectiveData,
    concept_to_indicator_mapping,
)
from sqlalchemy import create_engine

ENGINE = create_engine(f"sqlite:///{str(db_path)}", echo=False)


def insert_table(df, table_name):
    df.to_sql(table_name, con=ENGINE, if_exists="replace")


def create_indicator_table():
    df = pd.read_csv(south_sudan_data, index_col=False)
    insert_table(df, "indicator")


def create_dssat_data_table():
    state_dict = {"NBG":"Northern Bahr El Ghazal","Unity":"Unity"}
    crop_dict = {"MAIZ":"maize", "SORG":"sorghum"}
    dfs = []
    for filename in ("NBG_MAIZ", "NBG_SORG", "Unity_MAIZ", "Unity_SORG"):
        df = pd.read_csv(data_dir / "SSD_csv" / f"{filename}.csv", usecols=[0, 1, 2])
        state, crop = filename.split("_")
        df["State"], df["Crop"]  = state_dict[state], crop_dict[crop]
        df["Source"] = "DSSAT"
        dfs.append(df)

    insert_table(pd.concat(dfs), "dssat")


def create_adjectiveData_table():
    df = pd.read_csv(adjectiveData, index_col=False, delim_whitespace=True)
    insert_table(df, "gradableAdjectiveData")


def create_concept_to_indicator_mapping_table():
    df = pd.read_table(
        concept_to_indicator_mapping,
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
    df.to_csv('test.csv', index=False)

    insert_table(df, "concept_to_indicator_mapping")


if __name__ == "__main__":
    # create_dssat_data_table()
    # create_indicator_table()
    # create_adjectiveData_table()
    create_concept_to_indicator_mapping_table()
