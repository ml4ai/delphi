""" Get all preassembled statements, prune away unneeded information, pickle
them. """

import sys
import pickle
from delphi.utils.indra import get_statements_from_json_file

if __name__ == "__main__":
    all_sts = get_statements_from_json_file(sys.argv[1])

    with open(sys.argv[2], "wb") as f:
        pickle.dump(all_sts, f)
