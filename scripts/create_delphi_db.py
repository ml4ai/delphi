import sys
import numpy as np
import pandas as pd
from delphi.paths import (
    data_dir,
    south_sudan_data,
    adjectiveData,
    concept_to_indicator_mapping,
)
from sqlalchemy import create_engine

ENGINE = create_engine(f"sqlite:///{sys.argv[3]}", echo=False)

def insert_table(df, table_name):
    df.to_sql(table_name, con=ENGINE, if_exists="replace")


def create_indicator_table(indicator_table):
    df = pd.read_table(indicator_table, index_col=False)
    insert_table(df, "indicator")


def create_adjectiveData_table():
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

    insert_table(df, "concept_to_indicator_mapping")


if __name__ == "__main__":
    create_indicator_table(sys.argv[1])
    create_adjectiveData_table()
    create_concept_to_indicator_mapping_table(sys.argv[2])
