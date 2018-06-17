import os
import pandas as pd
from glob import glob
from future.utils import lmap
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from ruamel.yaml.scalarstring import DoubleQuotedScalarString as S
from ruamel.yaml import YAML
import urllib.request as request
from delphi.paths import data_dir
from pathlib import Path
import requests
import zipfile
import subprocess as sp
import contextlib

read_csv = partial(
    pd.read_csv,
    encoding="latin-1",
    usecols=lambda colName: all(
        map(
            lambda x: x not in colName, ["Code", "Flag", "Note", "ElementGroup"]
        )
    ),
)


def tqdm_reporthook(t):
    """Wraps tqdm instance.
    Don't forget to close() or __exit__()
    the tqdm instance once you're done with it (easiest using `with` syntax).
    Example
    -------
    >>> with tqdm(...) as t:
    ...     reporthook = my_hook(t)
    ...     urllib.urlretrieve(..., reporthook=reporthook)
    """
    last_b = [0]

    def update_to(b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b

    return update_to


def download_file(url: str, filename: str):
    print(f"Downloading {url} to {filename}")
    with tqdm(
        ncols=80,
        unit="bytes",
        unit_scale=True,
        unit_divisor=1024,
    ) as t:
        reporthook = tqdm_reporthook(t)
        request.urlretrieve(url, filename, reporthook)


def download_FAOSTAT_data():
    url = (
        "http://fenixservices.fao.org/faostat/static/bulkdownloads/FAOSTAT.zip"
    )
    filename = str(Path(data_dir) / "FAOSTAT.zip")
    download_file(url, filename)

faostat_zipfile = Path(data_dir) / "FAOSTAT.zip"
faostat_dir = Path(data_dir) / "FAOSTAT"

def extract_zipfile(file, dir):
    with zipfile.ZipFile(file) as zf:
        zf.extractall(dir)

def _change_directory(destination_directory):
    cwd = os.getcwd()
    os.chdir(destination_directory)
    try: yield
    except: pass
    finally: os.chdir(cwd)

cd = contextlib.contextmanager(_change_directory)

def sp_unzip(f):
    sp.call(['unzip', f])

def clean_FAOSTAT_data():

    if not (faostat_zipfile.is_file() or faostat_dir.is_dir()):
        download_FAOSTAT_data()
    if (faostat_zipfile.is_file() and not faostat_dir.is_dir()):
        os.mkdir(faostat_dir)
        with zipfile.ZipFile(str(faostat_zipfile)) as zf:
            zf.extractall(faostat_dir)
        with cd(str(faostat_dir.resolve())):
            zipfiles = glob("*.zip")
            with mp.Pool(mp.cpu_count()) as p:
                for _ in tqdm(p.imap_unordered(sp_unzip, zipfiles), total=len(zipfiles)):
                    pass

    dfs = []
    for filename in tqdm(glob(str(faostat_dir)+'/*.csv')):
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
    df.to_csv(str(Path(data_dir)/"south_sudan_data.csv"), index=False, sep="|")


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
    df = pd.read_csv("south_sudan_data.csv")
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
    # yaml.preserve_quotes=True
    yaml.default_flow_style = False

    with open("fao_variable_ontology.yml", "w") as f:
        yaml.dump(d, f)

if __name__ == "__main__":
    clean_FAOSTAT_data()
