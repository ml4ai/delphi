from delphi.utils import download_file
from delphi.paths import data_dir
from pathlib import Path

def download_WDI_data():
    url = "http://databank.worldbank.org/data/download/WDI_csv.zip"
    wdi_zipfile = Path(data_dir) / "WDI_csv.zip"
    if not wdi_zipfile.is_file():
        download_file(url, wdi_zipfile)

if __name__ == "__main__":
    download_WDI_data()
