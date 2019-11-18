import os
import sys
import zipfile
import calendar
from pathlib import Path
from glob import glob
import shapefile
from future.utils import lzip
from delphi.utils.shell import cd
from delphi.utils.web import download_file
from tqdm import tqdm
import matplotlib as mpl
import pandas as pd
import sys

mpl.rcParams["backend"] = "Agg"
from matplotlib import pyplot as plt
from shapely.geometry import Polygon, MultiPolygon


def process_FEWSNET_IPC_data(shpfile: str, title: str):
    admin_boundaries_shapefile = "data/raw/FEWS/FEWSNET_World_Admin/FEWSNET_Admin2"
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

    def fill_and_plot(points, color_code):
        xs, ys = lzip(*points)
        ax.plot(xs, ys, linewidth=0.5, color="grey")
        ax.fill(xs, ys, color=colors[color_code])

    fs_polygons = []

    for i, sr in tqdm(enumerate(sf.shapeRecords())):
        nparts = len(sr.shape.parts)
        parts, points = sr.shape.parts, sr.shape.points
        CS = int(sr.record[0])
        if nparts == 1:
            # fill_and_plot(points, CS)
            fs_polygons.append((Polygon(points), int(sr.record[0])))
        else:
            for ip, part in enumerate(parts):
                if ip < nparts - 1:
                    i1 = parts[ip + 1] - 1
                else:
                    i1 = len(points)
                # fill_and_plot(points[part : i1 + 1], CS),
                fs_polygons.append(
                    (Polygon(points[part : i1 + 1]), int(sr.record[0]))
                )

    south_sudan_srs = [
        sr for sr in sf_admin.shapeRecords() if sr.record[3] == "South Sudan"
    ]

    lines = []

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
                CS = int(fs_polygon[1])
                fill_and_plot(sr.shape.points, CS)
                lines.append(
                    "\t".join([str(x) for x in sr.record] + [str(CS)])
                )

    with open("ipc_data.tsv", "w") as f:
        f.write("\n".join(lines))

    plt.savefig("shape.pdf")


def get_polygons(shape):
    parts, points = shape.parts, shape.points
    nparts = len(parts)
    polygons = []

    if nparts == 1:
        polygons.append(Polygon(points))
    else:
        for ip, part in enumerate(parts):
            if ip < nparts - 1:
                i1 = parts[ip + 1] - 1
            else:
                i1 = len(points)
            polygons.append(Polygon(points[part : i1 + 1]))

    return polygons


def create_food_security_data_table(region: str, country: str):
    admin_boundaries_shapefile = "data/raw/FEWS/FEWSNET_World_Admin/FEWSNET_Admin2"
    sf_admin = shapefile.Reader(admin_boundaries_shapefile)
    south_sudan_srs = [
        x for x in sf_admin.shapeRecords() if x.record[3] == country
    ]

    path = f"data/raw/FEWS/ALL_HFIC/{region}"
    ipc_records = []
    with cd(path):
        shapefiles = glob("*.shp")
        for filename in tqdm(shapefiles, unit="shapefile"):
            year = int(filename[3:7])
            month = int(filename[7:9])
            reader = shapefile.Reader(filename)
            for i, fs_sr in tqdm(
                enumerate(reader.shapeRecords()),
                unit="Food security shapeRecord",
            ):
                parts, points = fs_sr.shape.parts, fs_sr.shape.points
                nparts = len(parts)
                CS = int(fs_sr.record[0])
                fs_polygons = get_polygons(fs_sr.shape)

                for sr in tqdm(south_sudan_srs, desc=f"{country} Counties"):
                    county_polygon = Polygon(sr.shape.points)
                    for fs_polygon in tqdm(
                        fs_polygons, unit="Food security polygon"
                    ):
                        if county_polygon.buffer(-0.05).intersects(fs_polygon):
                            ipc_records.append(
                                {
                                    "Country": sr.record[3],
                                    "State": sr.record[4],
                                    "County": sr.record[8],
                                    "Year": year,
                                    "Month": month,
                                    "Value": CS,
                                    "Variable": "IPC Phase Classification",
                                    "Unit": "IPC Phase",
                                    "Source": "FEWSNET",
                                }
                            )
    df = pd.DataFrame(ipc_records)
    df.to_csv(sys.argv[1], sep="\t", index=False)


if __name__ == "__main__":
    region = "East Africa"
    year = 2018
    month = 2
    month_str = "0" + str(month) if len(str(month)) < 2 else str(month)
    region_str = "".join([x[0] for x in region.split()])
    region_year_str = (
        f"{region_str}{year}" if year == 2017 else f"{region_str}_{year}"
    )
    shpfile = f"data/raw/FEWS/ALL_HFIC/{region}/{region_year_str}{month_str}_CS"
    title = "\n".join(
        (
            f"{region} Food Security Outcomes",
            f"{calendar.month_name[month]} {year}",
        )
    )
    #process_FEWSNET_IPC_data(shpfile, title)
    create_food_security_data_table("East Africa", "South Sudan")
