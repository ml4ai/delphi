import os
from pathlib import Path
import json

data_dir = Path(os.environ.get("DELPHI_DATA", ""))
db_path = Path(os.environ.get("DELPHI_DB", "postgres:delphi@localhost:5432/delphi"))

adjectiveData = data_dir / "adjectiveData.tsv"
south_sudan_data = data_dir / "south_sudan_data.csv"
concept_to_indicator_mapping = data_dir / "concept_to_indicator_mapping.txt"

os.environ["CLASSPATH"] = (
    str(Path(__file__).parent / "translators"/ "for2py" / "bin"/"*")
)
