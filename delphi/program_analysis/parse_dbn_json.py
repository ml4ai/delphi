import json
import pprint
import pathlib
from typing import Dict
import os


def read_dbn_from_json(dbn_json_source_file: str) -> Dict:
    """
    Read DBN from JSON file.
    :param dbn_json_source_file: JSON source filename in DBN.JSON format
    :return: Dict representing JSON
    """

    with open(dbn_json_source_file) as fin:
        dbn_json = json.load(fin)

    return dbn_json


if __name__ == '__main__':
    example_source_file = 'crop_yield_DBN.json'
    filepath = (pathlib.Path(__file__).parent / '..' / '..' /
                'data' / 'program_analysis' / 'pa_crop_yield_v0.2').resolve()
    dbn_json_source_file = os.path.join(filepath, example_source_file)
    data = read_dbn_from_json(dbn_json_source_file)
    print('DBN_JSON:')
    pprint.pprint(data)
