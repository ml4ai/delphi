import pandas as pd
from pathlib import Path
import sys

data_dir = Path("data")


def clean_UNHCR_data():
    """Data cleaning for United Nations High Commissioner for Refugees (UNHCR)
    monthly data for asylum seekers from South Sudan. The data is tracked by newly
    lodged asylum applications, thus it may not equate to all migration activities
    concerning South Sudan (obviously not IDPs/Internal Resettlement).
    Note: As suggested by being considered newly lodged applications,
    this count attempts to disclude repeat/re-opened applications.
    """
    # header=2 is required since the raw data has extra unneeded descriptions at
    # the top of the file
    df = pd.read_csv(
        "data/raw/unhcr_popstats_export_asylum_seekers_monthly_2019_06_24_192535.csv",
        header=2,
    )

    # As of right now (but will change soon) I am focused just on the origin of
    # the asylum seekers (South Sudan)
    df = df.drop(columns=["Country / territory of asylum/residence"])

    # Months are changed to numerical values 1-12 for easier sorting. I made
    # them floats to match the entries in south_sudan_data.tsv
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
    df.Month = df.Month.map(d)

    # Sorting values by year and month
    df = df.sort_values(by=["Year", "Month"])

    # Please Read: The * in the raw data represent values between 1 and 4 and
    # are only present in the 2017 and 2018 entries. The UNHCR's reasoning is
    # that with such few counts and being such recent data, the privacy of these
    # individuals could be easily compromised. In my cleaning, I have decided to
    # round-down from the median value between 1-4. Thus all * are being changed
    # to 2.
    df = df.replace("*", "")

    # Values are originally inferred as strings from the raw data because of the
    #'*'.
    df.Value = pd.to_numeric(df.Value)

    # Since I am focused on just a total count of Asylum Seekers from South
    # Sudan for each time point, we need to tally up the Asylum Seeker count
    # from multiple destinations for each time point.
    df["Value"] = df.groupby(["Year", "Month"])["Value"].transform("sum")
    df = df.drop_duplicates(subset=["Year", "Month"])

    # Rename Origin to just Country
    df.columns = ["Country", "Year", "Month", "Value"]

    # Add all needed axes. I choose to call the variable 'New asylum
    # seeking applicants' and the units 'applicants'.
    df["State"] = None
    df["County"] = None
    df["Source"] = "UNHCR"
    df["Unit"] = "applicants"
    df["Variable"] = "New asylum seeking applicants"

    # Reordered to fit south_sudan_data.tsv
    df = df.reindex(
        [
            "Country",
            "County",
            "Month",
            "Source",
            "State",
            "Unit",
            "Value",
            "Variable",
            "Year",
        ],
        axis=1,
    )

    df.to_csv(
        str(data_dir / "indicator_data_UNHCR.tsv"), index=False, sep="\t"
    )


if __name__ == "__main__":
    clean_UNHCR_data()
