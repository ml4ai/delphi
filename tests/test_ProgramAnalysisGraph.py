import json
from datetime import date
from conftest import *


def test_ProgramAnalysisGraph_init(PAG):
    assert PAG == CROP_YIELD_JSON_DICT


CROP_YIELD_JSON_DICT = {
    "functions": [
        {
            "name": "UPDATE_EST__assign__TOTAL_RAIN_0",
            "type": "assign",
            "target": "TOTAL_RAIN",
            "sources": [
                {"name": "TOTAL_RAIN", "type": "variable"},
                {"name": "RAIN", "type": "variable"},
            ],
            "body": [
                {
                    "type": "lambda",
                    "name": "UPDATE_EST__lambda__TOTAL_RAIN_0",
                    "reference": 5,
                }
            ],
        },
        {
            "name": "UPDATE_EST__condition__IF_1_0",
            "type": "assign",
            "target": "IF_1",
            "sources": [{"name": "TOTAL_RAIN", "type": "variable"}],
            "body": [
                {
                    "type": "lambda",
                    "name": "UPDATE_EST__lambda__IF_1_0",
                    "reference": 6,
                }
            ],
        },
        {
            "name": "UPDATE_EST__assign__YIELD_EST_0",
            "type": "assign",
            "target": "YIELD_EST",
            "sources": [{"name": "TOTAL_RAIN", "type": "variable"}],
            "body": [
                {
                    "type": "lambda",
                    "name": "UPDATE_EST__lambda__YIELD_EST_0",
                    "reference": 7,
                }
            ],
        },
        {
            "name": "UPDATE_EST__assign__YIELD_EST_1",
            "type": "assign",
            "target": "YIELD_EST",
            "sources": [{"name": "TOTAL_RAIN", "type": "variable"}],
            "body": [
                {
                    "type": "lambda",
                    "name": "UPDATE_EST__lambda__YIELD_EST_1",
                    "reference": 9,
                }
            ],
        },
        {
            "name": "UPDATE_EST__decision__YIELD_EST_0",
            "type": "assign",
            "target": "YIELD_EST",
            "sources": [
                {"name": "IF_1_0", "type": "variable"},
                {"name": "YIELD_EST_2", "type": "variable"},
                {"name": "YIELD_EST_1", "type": "variable"},
            ],
        },
        {
            "name": "UPDATE_EST",
            "type": "container",
            "input": [
                {"name": "RAIN", "domain": "real"},
                {"name": "TOTAL_RAIN", "domain": "real"},
                {"name": "YIELD_EST", "domain": "real"},
            ],
            "variables": [
                {"name": "TOTAL_RAIN", "domain": "real"},
                {"name": "RAIN", "domain": "real"},
                {"name": "IF_1", "domain": "boolean"},
                {"name": "YIELD_EST", "domain": "real"},
            ],
            "body": [
                {
                    "name": "UPDATE_EST__assign__TOTAL_RAIN_0",
                    "output": {"variable": "TOTAL_RAIN", "index": 1},
                    "input": [
                        {"variable": "TOTAL_RAIN", "index": 0},
                        {"variable": "RAIN", "index": 0},
                    ],
                },
                {
                    "name": "UPDATE_EST__condition__IF_1_0",
                    "output": {"variable": "IF_1", "index": 0},
                    "input": [{"variable": "TOTAL_RAIN", "index": 1}],
                },
                {
                    "name": "UPDATE_EST__assign__YIELD_EST_0",
                    "output": {"variable": "YIELD_EST", "index": 1},
                    "input": [{"variable": "TOTAL_RAIN", "index": 1}],
                },
                {
                    "name": "UPDATE_EST__assign__YIELD_EST_1",
                    "output": {"variable": "YIELD_EST", "index": 2},
                    "input": [{"variable": "TOTAL_RAIN", "index": 1}],
                },
                {
                    "name": "UPDATE_EST__decision__YIELD_EST_0",
                    "output": {"variable": "YIELD_EST", "index": 3},
                    "input": [
                        {"variable": "IF_1", "index": 0},
                        {"variable": "YIELD_EST", "index": 2},
                        {"variable": "YIELD_EST", "index": 1},
                    ],
                },
            ],
        },
        {
            "name": "CROP_YIELD__assign__MAX_RAIN_0",
            "type": "assign",
            "target": "MAX_RAIN",
            "sources": [],
            "body": {"type": "literal", "dtype": "real", "value": "4.0"},
        },
        {
            "name": "CROP_YIELD__assign__CONSISTENCY_0",
            "type": "assign",
            "target": "CONSISTENCY",
            "sources": [],
            "body": {"type": "literal", "dtype": "real", "value": "64.0"},
        },
        {
            "name": "CROP_YIELD__assign__ABSORPTION_0",
            "type": "assign",
            "target": "ABSORPTION",
            "sources": [],
            "body": {"type": "literal", "dtype": "real", "value": "0.6"},
        },
        {
            "name": "CROP_YIELD__assign__YIELD_EST_0",
            "type": "assign",
            "target": "YIELD_EST",
            "sources": [],
            "body": {"type": "literal", "dtype": "integer", "value": "0"},
        },
        {
            "name": "CROP_YIELD__assign__TOTAL_RAIN_0",
            "type": "assign",
            "target": "TOTAL_RAIN",
            "sources": [],
            "body": {"type": "literal", "dtype": "integer", "value": "0"},
        },
        {
            "name": "CROP_YIELD__assign__RAIN_0",
            "type": "assign",
            "target": "RAIN",
            "sources": [
                {"name": "DAY", "type": "variable"},
                {"name": "CONSISTENCY", "type": "variable"},
                {"name": "MAX_RAIN", "type": "variable"},
                {"name": "ABSORPTION", "type": "variable"},
            ],
            "body": [
                {
                    "type": "lambda",
                    "name": "CROP_YIELD__lambda__RAIN_0",
                    "reference": 25,
                }
            ],
        },
        {
            "name": "CROP_YIELD__loop_plate__DAY_0",
            "type": "loop_plate",
            "input": [
                "CONSISTENCY",
                "MAX_RAIN",
                "ABSORPTION",
                "RAIN",
                "TOTAL_RAIN",
                "YIELD_EST",
            ],
            "index_variable": "DAY",
            "index_iteration_range": {
                "start": {"type": "literal", "dtype": "integer", "value": 1},
                "end": {"value": 32, "dtype": "integer", "type": "literal"},
            },
            "body": [
                {
                    "name": "CROP_YIELD__assign__RAIN_0",
                    "output": {"variable": "RAIN", "index": 0},
                    "input": [
                        {"variable": "DAY", "index": -1},
                        {"variable": "CONSISTENCY", "index": -1},
                        {"variable": "MAX_RAIN", "index": -1},
                        {"variable": "ABSORPTION", "index": -1},
                    ],
                },
                {
                    "function": "UPDATE_EST",
                    "output": {},
                    "input": [
                        {"variable": "RAIN", "index": 0},
                        {"variable": "TOTAL_RAIN", "index": -1},
                        {"variable": "YIELD_EST", "index": -1},
                    ],
                },
                {
                    "function": "print",
                    "output": {},
                    "input": [
                        {"variable": "DAY", "index": -1},
                        {"variable": "YIELD_EST", "index": -1},
                    ],
                },
            ],
        },
        {
            "name": "CROP_YIELD",
            "type": "container",
            "input": [],
            "variables": [
                {"name": "DAY", "domain": "integer"},
                {"name": "RAIN", "domain": "real"},
                {"name": "YIELD_EST", "domain": "real"},
                {"name": "TOTAL_RAIN", "domain": "real"},
                {"name": "MAX_RAIN", "domain": "real"},
                {"name": "CONSISTENCY", "domain": "real"},
                {"name": "ABSORPTION", "domain": "real"},
            ],
            "body": [
                {
                    "name": "CROP_YIELD__assign__MAX_RAIN_0",
                    "output": {"variable": "MAX_RAIN", "index": 2},
                    "input": [],
                },
                {
                    "name": "CROP_YIELD__assign__CONSISTENCY_0",
                    "output": {"variable": "CONSISTENCY", "index": 2},
                    "input": [],
                },
                {
                    "name": "CROP_YIELD__assign__ABSORPTION_0",
                    "output": {"variable": "ABSORPTION", "index": 2},
                    "input": [],
                },
                {
                    "name": "CROP_YIELD__assign__YIELD_EST_0",
                    "output": {"variable": "YIELD_EST", "index": 2},
                    "input": [],
                },
                {
                    "name": "CROP_YIELD__assign__TOTAL_RAIN_0",
                    "output": {"variable": "TOTAL_RAIN", "index": 2},
                    "input": [],
                },
                {
                    "name": "CROP_YIELD__loop_plate__DAY_0",
                    "inputs": [
                        "CONSISTENCY",
                        "MAX_RAIN",
                        "ABSORPTION",
                        "RAIN",
                        "TOTAL_RAIN",
                        "YIELD_EST",
                    ],
                    "output": {},
                },
                {
                    "function": "print",
                    "output": {},
                    "input": [{"variable": "YIELD_EST", "index": 2}],
                },
            ],
        },
    ],
    "start": "CROP_YIELD",
    "name": "crop_yield.json",
    "dateCreated": str(date.today()),
}
