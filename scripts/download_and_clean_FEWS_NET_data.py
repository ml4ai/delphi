from delphi.utils import download_file, cd
from delphi.paths import data_dir
from pathlib import Path

def download_FEWS_NET_data():
    url = "http://shapefiles.fews.net.s3.amazonaws.com/ALL_HFIC.zip"
    download_file(url, Path(data_dir) / url.split('/')[-1])

if __name__ == '__main__':
    download_FEWS_NET_data()
