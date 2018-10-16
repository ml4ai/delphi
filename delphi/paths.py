import os
from pathlib import Path
import json

data_dir = str(Path(os.environ["DELPHI_DATA"]).resolve())
adjectiveData = str((Path(data_dir) / "adjectiveData.tsv").resolve())
south_sudan_data = str((Path(data_dir) / "south_sudan_data.csv").resolve())
concept_to_indicator_mapping = str(
    (Path(data_dir) / "concept_to_indicator_mapping.txt").resolve()
)
