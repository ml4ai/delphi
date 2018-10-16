import os
from pathlib import Path
import json

with open("../delphi_config.json", "r") as f:
    config = json.load(f)

data_dir = str(Path(config["delphi_data_path"].replace("~", os.path.expanduser("~"))).resolve())
adjectiveData = str((Path(data_dir) / "adjectiveData.tsv").resolve())
south_sudan_data = str((Path(data_dir) / "south_sudan_data.csv").resolve())
concept_to_indicator_mapping = str(
    (Path(data_dir) / "concept_to_indicator_mapping.txt").resolve()
)
