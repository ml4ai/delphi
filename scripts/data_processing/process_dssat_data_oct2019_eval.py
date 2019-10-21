import sys
import pandas as pd
from glob import glob
import numpy as np

dfs = list()

filenames = glob("data/raw/data_for_november_2019_evaluation/*/dssat/*.csv")

for filename in filenames:
    df = pd.read_csv(
        filename, encoding="latin-1", usecols=["HWAMA", "TAVGA", "PRCPA"]
    )
    df["Year"] = np.arange(2018 - len(df), 2018)
    big_frame = pd.DataFrame(
        {
            "Value": df.iloc[:, 0].values,
            "Variable": df.columns[0],
            "Year": df["Year"].values,
        }
    )

    for col in df.columns[1:-1]:
        val = df[col].values
        indicator = col
        df_new = pd.DataFrame({"Value": val, "Variable": indicator})
        df_new = pd.concat([df_new, df["Year"]], axis=1, join="inner")
        big_frame = pd.concat(
            [big_frame, df_new], sort=False, ignore_index=True
        )
        if "gambela" in filename:
            big_frame["State"] = "Gambela"
        elif "jonglei" in filename:
            big_frame["State"] = "Jonglei"
        else:
            big_frame["State"] = None

        if "SSD" in filename:
            big_frame["Country"] = "South Sudan"
        elif "ETH" in filename:
            big_frame["Country"] = "Ethiopia"
        else:
            raise Exception("Neither SSD or ETH in filename - are you sure you"
                    " are passing in the correct files?")

    dfs.append(big_frame)

big_frame = pd.concat(dfs, ignore_index=True, sort=True)

big_frame["Month"], big_frame["County"], big_frame["Unit"] = None, None, None
big_frame.loc[big_frame["Variable"] == "HWAMA", ["Unit"]] = "kg/ha"
big_frame.loc[big_frame["Variable"] == "PRCPA", ["Unit"]] = "mm"
big_frame.loc[big_frame["Variable"] == "TAVGA", ["Unit"]] = "Celsius"

dict_var = {
    "HWAMA": "Average Harvested Weight at Maturity (Maize)",
    "TAVGA": "Average Temperature",
    "PRCPA": "Average Precipitation",
}

big_frame["Variable"].replace(dict_var, inplace=True)

big_frame.to_csv(sys.argv[1], index=False, sep="\t")
