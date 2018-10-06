import os
import zipfile
from pathlib import Path
from delphi.utils import download_file, cd
from delphi.paths import data_dir

def download_FEWS_NET_data():
    url = "http://shapefiles.fews.net.s3.amazonaws.com/ALL_HFIC.zip"
    zipfile_name = Path(data_dir) / url.split('/')[-1]
    FEWS_NET_data_dir = Path(data_dir) / "FEWS_NET"
    download_file(url, zipfile_name)
    with zipfile.ZipFile(zipfile_name) as zf:
        zf.extractall(data_dir)

if __name__ == '__main__':
    download_FEWS_NET_data()
