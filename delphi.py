#!/usr/bin/env python
""" Main CLI Interface for Delphi

Usage:
    delphi.py  -h | --help
    delphi.py  <filename> [--grounding_threshold = <gt>]

Options:
    -h --help     Show this help message.
    --grounding_threshold = <gt>  [default: 1.0]
"""

import os
from delphi import app
from typing import Optional
from delphi.types import State
from indra.sources import eidos
from indra.statements import Influence, Agent
from indra.assemblers import CAGAssembler
import json
import docopt

def add_statements(state: State, statements,
        grounding_threshold: Optional[float]) -> State:
    """ Add statements to delphi session """
    state.statements = statements
    cag_assembler = CAGAssembler(state.statements)
    state.CAG = cag_assembler.make_model(grounding_threshold=grounding_threshold)
    state.elementsJSON = cag_assembler.export_to_cytoscapejs()
    state.elementsJSONforJinja = json.dumps(state.elementsJSON) 
    return state

if __name__ == '__main__':
    args = docopt.docopt(__doc__)

    ep = eidos.process_json_ld_file(args['<filename>'])
    app.state = add_statements(app.state, ep.statements,
            grounding_threshold=float(args['--grounding_threshold']))
    app.run()
