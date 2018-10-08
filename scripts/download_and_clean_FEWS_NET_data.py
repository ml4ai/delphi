import os
import zipfile
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


def download_FEWS_NET_data():
    url = "http://shapefiles.fews.net.s3.amazonaws.com/ALL_HFIC.zip"
    zipfile_name = Path(data_dir) / url.split("/")[-1]
    FEWS_NET_data_dir = Path(data_dir) / "FEWS_NET"
    download_file(url, zipfile_name)
    with zipfile.ZipFile(zipfile_name) as zf:
        zf.extractall(data_dir)


def process_FEWS_NET_data(shpfile: str):
    # read the shapefile
    colors = {
        0: "white",
        1: "lightgreen",
        2: "yellow",
        3: "orange",
        4: "red",
        5: "brown",
        88: "white",
    }
    sf = shapefile.Reader(shpfile)
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_aspect("equal")
    plt.style.use("ggplot")

    def fill_and_plot(points, sr):
        xs, ys = lzip(*points)
        ax.plot(xs, ys, linewidth=0.5, color="grey")
        ax.fill(xs, ys, color=colors[int(sr.record[0])])

    for sr in tqdm(sf.iterShapeRecords()):
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
    # download_FEWS_NET_data()
    shpfile = "data/ALL_HFIC/East Africa/EA201702_CS"
    process_FEWS_NET_data(shpfile)
