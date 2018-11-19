#!/usr/bin/env python3

"""
File: f2py_pp.py
Author: Saumya Debray
Purpose: Preprocess Fortran source files prior to parsing to fix up
        some constructs (such as continuation lines) that are
        problematic for the OpenFortranParser front end.
Usage:
        f2py_pp  infile  outfile

            outfile is the cleaned-up version of infile
"""

import sys
from delphi.program_analysis.autoTranslate.scripts.fortran_syntax import *

def process(infile, outfile):
    with open(infile, mode="r", encoding="latin-1") as f:
        # remove lines that are entirely comments
        lines = [line for line in f if not line_is_comment(line)]

        # remove partial-line comments
        lines = [rm_trailing_comment(line) for line in lines]


    # merge continuation lines
    chg = True
    while chg:
        chg = False
        i = 0
        while i < len(lines):
            line = lines[i]
            llstr = line.lstrip()
            if len(llstr) > 0 and llstr[0] == "&":  # continuation character
                prevline = lines[i - 1]
                line = llstr[1:].lstrip()
                prevline = prevline.rstrip() + line
                lines[i - 1] = prevline
                lines.pop(i)
                chg = True
            i += 1

    with open(outfile, "w") as f:
        f.write("".join(lines))


def rm_trailing_comment(line: str) -> str:
    """rm_trailing_comment(line) takes a line and returns the line with any
    trailing comment (the '!' comment character and subsequent characters
    to the end of the line) removed."""

    if line.find("!") == -1:
        return line

    i = 0
    while i < len(line):
        if line[i] == "'":
            j = line.find("'", i + 1)
            if j == -1:
                sys.stderr.write("WEIRD: unbalanced quote ': line = " + line)
                return line
            else:
                i = j + 1
        elif line[i] == '"':
            j = line.find('"', i + 1)
            if j == -1:
                sys.stderr.write('WEIRD: unbalanced quote ": line = ' + line)
                return line
            else:
                i = j + 1
        elif line[i] == "!":  # partial-line comment
            return line[:i].rstrip() + "\n"
        else:
            i += 1

    return line


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.stderr.write("*** USAGE: f2py_pp.py <infile> <outfile>\n")
        sys.exit(1)

    infile, outfile = sys.argv[1], sys.argv[2]
    process(infile, outfile)
