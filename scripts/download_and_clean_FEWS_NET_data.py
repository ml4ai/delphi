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
from shapely.geometry import Polygon, MultiPolygon


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
    admin_boundaries_shapefile = "data/FEWSNET_World_Admin/FEWSNET_Admin2"
    sf_admin = shapefile.Reader(admin_boundaries_shapefile)
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

    fs_polygons = []
    for i, sr in tqdm(enumerate(sf.shapeRecords())):
        nparts = len(sr.shape.parts)
        parts, points = sr.shape.parts, sr.shape.points
        if nparts == 1:
            # fill_and_plot(points, sr)
            fs_polygons.append((Polygon(points), int(sr.record[0])))
        else:
            for ip, part in enumerate(parts):
                if ip < nparts - 1:
                    i1 = parts[ip + 1] - 1
                else:
                    i1 = len(points)
                # fill_and_plot(points[part : i1 + 1], sr),
                fs_polygons.append(
                    (Polygon(points[part : i1 + 1]), int(sr.record[0]))
                )
    south_sudan_srs = [
        sr for sr in sf_admin.shapeRecords() if sr.record[3] == "South Sudan"
    ]
    for sr in tqdm(south_sudan_srs, desc="South Sudan Counties"):
        county_polygon = Polygon(sr.shape.points)
        for fs_polygon in tqdm(fs_polygons, desc="fs_polygons"):
            if county_polygon.buffer(-0.05).intersects(fs_polygon[0]):
                centroid = county_polygon.centroid
                ax.text(
                    centroid.x,
                    centroid.y,
                    sr.record[8],
                    fontsize=6,
                    horizontalalignment="center",
                )
                xs, ys = lzip(*sr.shape.points)
                ax.plot(xs, ys, linewidth=0.5, color="grey")
                ax.fill(
                    xs, ys, linewidth=0.5, color=colors[int(fs_polygon[1])]
                )
                break

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
