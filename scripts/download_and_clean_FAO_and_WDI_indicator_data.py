import os
import pandas as pd
from glob import glob
from future.utils import lmap
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from ruamel.yaml.scalarstring import DoubleQuotedScalarString as S
from ruamel.yaml import YAML
from delphi.paths import data_dir
from delphi.utils.shell import cd
from delphi.utils.web import download_file
from pathlib import Path
import zipfile
import subprocess as sp
import contextlib


read_csv = partial(
    pd.read_csv,
    encoding="latin-1",
    usecols=lambda colName: all(
        map(
            lambda x: x not in colName,
            ["Code", "Flag", "Note", "ElementGroup"],
        )
    ),
)


def download_FAOSTAT_data(url, faostat_zipfile):
    """ Download data from FAOSTAT database """
    download_file(url, str(faostat_zipfile))


def sp_unzip(f):
    sp.call(["unzip", f])


def clean_FAOSTAT_data(faostat_zipfile, faostat_dir):

    if not (faostat_zipfile.is_file() or faostat_dir.is_dir()):
        download_FAOSTAT_data()
    if faostat_zipfile.is_file() and not faostat_dir.is_dir():
        os.mkdir(faostat_dir)
        with zipfile.ZipFile(str(faostat_zipfile)) as zf:
            zf.extractall(faostat_dir)
        with cd(str(faostat_dir.resolve())):
            zipfiles = glob("*.zip")
            with mp.Pool(mp.cpu_count()) as p:
                for _ in tqdm(
                    p.imap_unordered(sp_unzip, zipfiles), total=len(zipfiles)
                ):
                    pass

    dfs = []
    for filename in tqdm(glob(str(faostat_dir) + "/*.csv")):
        df = read_csv(filename)
        df = df.rename(columns={"Country": "Area", "Months": "Month"})
        if "Currency" in df.columns:
            df = df.rename(columns={"Currency": "Item", "Item": "Element"})
        if "Reporter Countries" in df.columns:
            df = df.rename(columns={"Reporter Countries": "Area"})
        if "Survey" in df.columns:
            df = df.rename(
                columns={
                    "Breadown by Sex of the Household Head": "Sex of the Household Head",
                    "Indicator": "Item",
                    "Measure": "Element",
                }
            )
            df["Area"] = df["Survey"].str.split().str.get(0)
            df["Year"] = df["Survey"].str.split().str.get(-1)
            del df["Survey"]
            df = df[df["Sex of the Household Head"] == "Total"]
            df = df[df["Breakdown Variable"] == "Country-level"]
            del df["Sex of the Household Head"]
            del df["Breakdown Variable"]
        if "Donor Country" in df.columns:
            df = df.rename(columns={"Recipient Country": "Area"})
            del df["Donor Country"]
            df["Element"] = "Food aid shipments"
        if set(df.columns.values) == set(
            ["Area", "Item", "Element", "Year", "Unit", "Value"]
        ):
            df = df[df["Area"] == "South Sudan"]
            del df["Area"]
            dfs.append(df)

        df["filename"] = filename.split("/")[-1]

    df = pd.concat(dfs)
    df.to_csv(
        str(Path(data_dir) / "south_sudan_data_fao.csv"), index=False, sep="|"
    )


def process_variable_name(k, e):
    if "," in e:
        spstrs = e.split(",")
        if len(e.split()) == 2:
            if spstrs[1].endswith("s"):
                spstrs[1] = spstrs[1].rstrip("s")
            return S((" ".join([k, spstrs[1].lstrip(), spstrs[0]]).lower()))
        else:
            return S(" ".join([k, e]))
    else:
        return S(" ".join([k, e]))


def construct_FAO_ontology():
    """ Construct FAO variable ontology for use with Eidos. """
    df = pd.read_csv("south_sudan_data_fao.csv")
    gb = df.groupby("Element")

    d = [
        {
            "events": [
                {
                    k: [
                        {e: [process_variable_name(k, e)]}
                        for e in list(set(gb.get_group(k)["Item"].tolist()))
                    ]
                }
                for k in gb.groups.keys()
            ]
        }
    ]

    yaml = YAML()
    yaml.default_flow_style = False

    with open("fao_variable_ontology.yml", "w") as f:
        yaml.dump(d, f)


def download_WDI_data(wdi_zipfile, wdi_data_dir):
    url = "http://databank.worldbank.org/data/download/WDI_csv.zip"
    if not wdi_zipfile.is_file():
        download_file(url, wdi_zipfile)
    if not wdi_data_dir.is_dir():
        os.mkdir(wdi_data_dir)
        with zipfile.ZipFile(str(wdi_zipfile)) as zf:
            zf.extractall(wdi_data_dir)


def clean_WDI_data(wdi_data_dir):
    df = pd.read_csv(
        wdi_data_dir / "WDIData.csv",
        usecols=[
            "Country Name",
            "Indicator Name",
            "2012",
            "2013",
            "2014",
            "2015",
            "2016",
            "2017",
        ],
    )
    df = df[df["Country Name"] == "South Sudan"]
    del df["Country Name"]
    df.to_csv(
        Path(data_dir) / "south_sudan_data_wdi.csv", index=False, sep="|"
    )


def combine_FAO_and_WDI_data():
    fao_df = pd.read_csv(Path(data_dir) / "south_sudan_data_fao.csv", sep="|")
    wdi_df = pd.read_csv(Path(data_dir) / "south_sudan_data_wdi.csv", sep="|")

    fao_df["Indicator Name"] = fao_df["Element"] + ", " + fao_df["Item"]

    del fao_df["Element"]
    del fao_df["Item"]

    wdi_df["Unit"] = (
        wdi_df["Indicator Name"].str.partition("(")[2].str.partition(")")[0]
    )
    wdi_df["Indicator Name"] = wdi_df["Indicator Name"].str.partition("(")[0]
    wdi_df = wdi_df.set_index(["Indicator Name", "Unit"])
    fao_df = (
        fao_df.pivot_table(
            values="Value", index=["Indicator Name", "Unit"], columns="Year"
        )
        .reset_index()
        .set_index(["Indicator Name", "Unit"])
    )

    df = pd.concat([fao_df, wdi_df], sort=True)

    # If a column name is something like 2010-2012, we make copies of its data for
    # three years - 2010, 2011, 2012

    for c in df.columns:
        if "-" in c:
            years = c.split("-")
            for y in range(int(years[0]), int(years[-1]) + 1):
                y = str(y)
                df[y] = df[y].fillna(df[c])

    df.to_csv(Path(data_dir) / "south_sudan_data.csv", sep="|")


if __name__ == "__main__":

    wdi_zipfile = Path(data_dir) / "WDI_csv.zip"
    wdi_data_dir = Path(data_dir) / "WDI"

    download_WDI_data(wdi_zipfile, wdi_data_dir)
    clean_WDI_data(wdi_data_dir)

    faostat_url = (
        "http://fenixservices.fao.org/faostat/static/bulkdownloads/FAOSTAT.zip"
    )
    faostat_zipfile = Path(data_dir) / "FAOSTAT.zip"
    faostat_dir = Path(data_dir) / "FAOSTAT"

    download_FAOSTAT_data(faostat_url, faostat_zipfile)
    clean_FAOSTAT_data(faostat_zipfile, faostat_dir)
