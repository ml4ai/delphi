#!/usr/bin/env python
""" Main CLI Interface for Delphi

Usage:
    delphi.py  -h | --help

Options:
    -h --help     Show this help message.
    --grounding_threshold = <gt>
"""

import os
from delphi import app
from typing import Optional
from indra.sources import eidos
from indra.assemblers import CAGAssembler
from delphi.types import State
from delphi.utils import flatMap
from future.utils import lfilter
from glob import glob
from tqdm import tqdm
import json
import docopt

if __name__ == '__main__':
    args = docopt.docopt(__doc__)
