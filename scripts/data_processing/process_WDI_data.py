import os, sys
import pandas as pd
from glob import glob
from future.utils import lmap
from tqdm import tqdm
import multiprocessing as mp
from functools import partial
from ruamel.yaml.scalarstring import DoubleQuotedScalarString as S
from ruamel.yaml import YAML
from delphi.utils.shell import cd
from delphi.utils.web import download_file
from pathlib import Path
import zipfile
import subprocess as sp
import contextlib

def clean_WDI_data(inputFile, outputFile):
    print("Cleaning WDI data")
    df = pd.read_csv(inputFile,
            usecols=[
            "Country Name",
            "Indicator Name"]+[str(x) for x in range(2012,2019)]
        )
    df.rename(
        columns={"Indicator Name": "Variable", "Country Name": "Country"},
        inplace=True,
    )
    df = df.query("Country == 'South Sudan' or Country == 'Ethiopia'")
    df.to_csv(outputFile, index=False, sep="\t")

if __name__ == "__main__":
    clean_WDI_data(sys.argv[1], sys.argv[2])
