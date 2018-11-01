#!/usr/bin/env python3

"""
    File: f2py-pp.py
    Author: Saumya Debray
    Purpose: Preprocess Fortran source files prior to parsing to fix up
            some constructs (such as continuation lines) that are
            problematic for the OpenFortranParser front end.
    Usage:
            f2py-pp  infile  outfile

                outfile is the cleaned-up version of infile
"""

import sys

def process(infile, outfile):
    # remove lines that are entirely comments
    lines = [line for line in infile if (line[0] == ' ' or line[0] == '\t')]

    #    # remove partial-line comments
    #    for i in range(len(lines)):
    #        line = lines[i]
    #        idx = line.find('!')
    #        if idx >= 0:
    #            lines[i] = line[:idx] + '\n'

    # merge continuation lines
    chg = True
    while chg:
        chg = False
        for i in range(len(lines)):
            # len(lines) may have changed because of deletions
            if i == len(lines):
                break
    
            line = lines[i]
            llstr = line.lstrip()
            if len(llstr) > 0 and llstr[0] == "&":     # continuation character
                prevline = lines[i-1]
                line = llstr[1:].lstrip()
                prevline = rm_trailing_comment(prevline).rstrip() + line
                lines[i-1] = prevline
                lines.pop(i)
                chg = True

    outline = ''.join(lines)
    outfile.write(outline)


# rm_trailing_comment(line) takes a line and returns the line with any
# trailing comment (the '!' comment character and subsequent characters
# to the end of the line) removed.
def rm_trailing_comment(line):
    if line.find('!') == -1:
        return line

    i = 0
    while i < len(line):
        if line[i] == "'":
            j = line.find("'", i+1)
            if j == -1:
                sys.stderr.write('WEIRD: unbalanced quote \': line = ' + line)
                return line
            else:
                i = j+1
        elif line[i] == '"':
            j = line.find('"', i+1)
            if j == -1:
                sys.stderr.write('WEIRD: unbalanced quote ": line = ' + line)
                return line
            else:
                i = j+1
        elif line[i] == '!':    # partial-line comment
            return line[:i]
        else:
            i += 1

    return line


def main():
    if len(sys.argv) < 3:
        sys.stderr.write('*** USAGE: f2py-pp.py <infile> <outfile>\n')
        sys.exit(1)

    infilename, outfilename = sys.argv[1], sys.argv[2]

    try:
        infile = open(infilename, mode="r", encoding="latin-1")
        outfile = open(outfilename, 'w')
    except IOError:
        sys.stderr.write('*** ERROR: could not open input/output files\n')
        sys.exit(1)

    process(infile, outfile)

    infile.close()
    outfile.close()

main()
