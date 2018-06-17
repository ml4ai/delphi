import os
from delphi.utils import download_file
from delphi.paths import data_dir
from pathlib import Path
import zipfile
import pandas as pd

wdi_zipfile = Path(data_dir) / "WDI_csv.zip"
wdi_data_dir = Path(data_dir) / "WDI"

def download_WDI_data():
    url = "http://databank.worldbank.org/data/download/WDI_csv.zip"
    if not wdi_zipfile.is_file():
        download_file(url, wdi_zipfile)
    if not wdi_data_dir.is_dir():
        os.mkdir(wdi_data_dir)
        with zipfile.ZipFile(str(wdi_zipfile)) as zf:
            zf.extractall(wdi_data_dir)

def clean_WDI_data():
    df = pd.read_csv(wdi_data_dir / 'WDIData.csv',
              usecols = ['Country Name', 'Indicator Name',
                         '2012', '2013', '2014', '2015', '2016', '2017'])
    df = df[df['Country Name'] == 'South Sudan']
    del df['Country Name']
    df.to_csv(Path(data_dir)/'south_sudan_data_wdi.csv', index=False, sep='|')

if __name__ == "__main__":
    download_WDI_data()
    clean_WDI_data()
