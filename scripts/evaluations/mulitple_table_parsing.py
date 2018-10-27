import re
import os
from itertools import takewhile, dropwhile, groupby, tee, chain, zip_longest
import pandas as pd
from pprint import pprint


def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks
    (from https://docs.python.org/3/library/itertools.html)

    grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"

    """
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def process_file(filename):
    df = pd.read_csv(filename)

    # Get the table headers
    first_column_entries = list(df.iloc[:, 0])
    table_headers = [
        entry
        for entry in first_column_entries
        if first_column_entries.count(entry) == 1
    ]

    # Define a grouping key function to split the CSV by the header rows
    grouping_key_function = lambda _tuple: _tuple[1].iloc[0] in table_headers
    key_group_tuples = groupby(df.iterrows(), grouping_key_function)
    groups = [
        [x[1].values.tolist() for x in list(group)]
        for k, g in key_group_tuples
    ]

    group_dict = {k[0][0]: g for k, g in grouper(groups, 2)}

    dfs = []
    for k, v in group_dict.items():
        df = pd.DataFrame(v)
        df.set_index(0, inplace=True)
        df.index = [f"{k} {i}" for i in df.index]
        df = df.stack().reset_index(name="Value")
        df.columns = ["Variable", "Month", "Value"]
        df["Units"] = "%"
        df["Country"] = "South Sudan"
        df["State"] = " ".join(
            re.findall("[A-Z][^A-Z]*", filename.split("_")[-2])
        )
        dfs.append(df)

    return pd.concat(dfs)


if __name__ == "__main__":
    FILENAME = os.environ["DELPHI_DATA"] + (
        "evaluations/12_month/"
        "Climis South Sudan Livestock Data"
        "/Livestock Diseases/LivestockDiseases_EasternEquatoria_2017.csv"
    )

    df = process_file(FILENAME)
    print(df)
