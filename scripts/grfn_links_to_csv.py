"""
This program allows a user to generate a CSV file with all of the grounding link
hypotheses from a GrFN JSON file. Note that this GrFN JSON file must already be
populated with link information.


Example execution:
    $ python grfn_links_to_csv.py <path-to-GrFN_with_groundings.json>

Author: Paul D. Hein
"""

import json
import sys
import csv

from delphi.GrFN.linking import make_link_tables


def main():
    filepath = sys.argv[1]
    grfn_with_hypotheses = json.load(open(filepath, "r"))

    link_tables = make_link_tables(grfn_with_hypotheses)
    outpath = filepath.replace("_with_groundings.json", "--links.csv")
    with open(outpath, "w", newline="") as csvfile:
        link_writer = csv.writer(csvfile, dialect="excel")

        for (_, scope, name, idx), var_data in link_tables.items():
            link_writer.writerow([scope, name, idx])
            link_writer.writerow(
                [
                    "Link Score",
                    "Var-Comm Score",
                    "Comm-Text Score",
                    "Text-Eqn Score",
                    "Comment Span",
                    "Text Mention",
                    "Equation Symbol",
                ]
            )
            for link_data in var_data:
                link_writer.writerow(
                    [
                        link_data["link_score"],
                        link_data["vc_score"],
                        link_data["ct_score"],
                        link_data["te_score"],
                        link_data["comm"],
                        link_data["txt"],
                        link_data["eqn"],
                    ]
                )
            link_writer.writerow([])


if __name__ == "__main__":
    main()
