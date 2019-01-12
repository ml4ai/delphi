import numpy as np
import pandas as pd
from delphi.paths import (
    db_path,
    south_sudan_data,
    adjectiveData,
    concept_to_indicator_mapping,
)
from sqlalchemy import create_engine

ENGINE = create_engine(f"sqlite:///{str(db_path)}", echo=False)


def create_indicator_table():
    df = pd.read_csv(south_sudan_data, index_col=False)
    df.to_sql("indicator", con=ENGINE, if_exists="replace")


def create_adjectiveData_table():
    df = pd.read_csv(adjectiveData, index_col=False, delim_whitespace=True)
    df.to_sql("gradableAdjectiveData", con=ENGINE, if_exists="replace")


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

    df.to_sql("concept_to_indicator_mapping", con=ENGINE, if_exists="replace")


if __name__ == "__main__":
    create_indicator_table()
    create_adjectiveData_table()
    create_concept_to_indicator_mapping_table()
