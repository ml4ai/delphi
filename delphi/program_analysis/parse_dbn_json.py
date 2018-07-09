import json
import pprint
import pathlib
from typing import Dict
import os

from pygraphviz import AGraph

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
    A = AGraph(directed=True)
    A.node_attr['shape'] = 'rectangle'
    for k in data['functions']:
        if k.get('sources') is not None:
            for s in k['sources']:
                A.add_node(s, shape='ellipse')
                A.add_edge(s, k['name'])
        if k.get('target') is not None:
            A.add_node(k['target'], shape='ellipse')
            A.add_edge(k['name'], k['target'])
    A.graph_attr['rankdir'] = 'LR'
    A.node_attr['fontname'] = 'Gill Sans'
    A.node_attr['style'] = 'rounded'
    A.draw('graph.pdf', prog='dot')
    # print('DBN_JSON:')
    # pprint.pprint(data)
