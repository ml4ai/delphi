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
from delphi.types import State
from indra.statements import Influence, Agent
from indra.assemblers import CAGAssembler
from indra.sources import eidos
import json
import docopt

def add_statements(state: State, statements) -> State:
    """ Add statements to delphi session """
    state.statements = statements
    cag_assembler = CAGAssembler(state.statements)
    state.CAG = cag_assembler.make_model()
    state.elementsJSON = cag_assembler.export_to_cytoscapejs()
    state.elementsJSONforJinja = json.dumps(state.elementsJSON) 
    return state


if __name__ == '__main__':
    args = docopt.docopt(__doc__)
    ep = eidos.process_json_ld_file('example_output.jsonld',
            grounding_threshold = float(args['--grounding_threshold']))

    app.state = add_statements(app.state, ep.statements)
    app.run()

