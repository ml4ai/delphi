import os
import zipfile
import calendar
from pathlib import Path
import shapefile
from future.utils import lzip
from delphi.utils import download_file, cd
from delphi.paths import data_dir
from tqdm import tqdm
import matplotlib as mpl

mpl.rcParams["backend"] = "Agg"
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon, PathPatch
from matplotlib.collections import PatchCollection


def download_FEWS_NET_admin_boundaries_data():
    url = "http://shapefiles.fews.net.s3.amazonaws.com/ADMIN/FEWSNET_World_Admin.zip"
    zipfile_name = Path(data_dir) / url.split("/")[-1]
    download_file(url, zipfile_name)
    directory = Path(data_dir) / (url.split("/")[-1].split(".")[0])
    os.makedirs(directory, exist_ok=True)
    with zipfile.ZipFile(zipfile_name) as zf:
        zf.extractall(directory)


def download_FEWS_NET_IPC_data():
    url = "http://shapefiles.fews.net.s3.amazonaws.com/ALL_HFIC.zip"
    zipfile_name = Path(data_dir) / url.split("/")[-1]
    download_file(url, zipfile_name)
    with zipfile.ZipFile(zipfile_name) as zf:
        zf.extractall(data_dir)


def process_FEWS_NET_IPC_data(shpfile: str, title: str):
    # read the shapefile
    colors = {
        0: "white",
        1: "#c3e2c3",
        2: "#f3e838",
        3: "#eb7d24",
        4: "#cd2026",
        5: "#5d060c",
        66: "aqua",
        88: "white",
        99: "white",
    }
    sf = shapefile.Reader(shpfile)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect("equal")
    ax.set_title(title)
    plt.style.use("ggplot")

    def fill_and_plot(points, sr):
        xs, ys = lzip(*points)
        ax.plot(xs, ys, linewidth=0.5, color="grey")
        ax.fill(xs, ys, color=colors[int(sr.record[0])])

    for i, sr in tqdm(
        enumerate(sf.iterShapeRecords()), total=len(sf.shapes())
    ):
        if len(sr.shape.parts) == 1:
            fill_and_plot(sr.shape.points, sr)
        else:
            for ip, part in enumerate(sr.shape.parts):
                if ip < len(sr.shape.parts) - 1:
                    i1 = sr.shape.parts[ip + 1] - 1
                else:
                    i1 = len(sr.shape.points)
                fill_and_plot(sr.shape.points[part : i1 + 1], sr),
    plt.savefig("shape.pdf")


if __name__ == "__main__":
    # download_FEWS_NET_IPC_data()
    # download_FEWS_NET_admin_boundaries_data()
    region = "East Africa"
    year = 2018
    month = 2
    process_month = (
        lambda month: "0" + str(month) if len(str(month)) < 2 else str(month)
    )
    month_str = process_month(month)
    region_str = "".join([x[0] for x in region.split()])
    region_year_str = (
        f"{region_str}{year}" if year == 2017 else f"{region_str}_{year}"
    )
    shpfile = f"data/ALL_HFIC/{region}/{region_year_str}{month_str}_CS"
    title = "\n".join(
        (
            f"{region} Food Security Outcomes",
            f"{calendar.month_name[month]} {year}",
        )
    )
    process_FEWS_NET_IPC_data(shpfile, title)
