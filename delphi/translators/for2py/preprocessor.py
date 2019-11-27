"""
This module implements functions to preprocess Fortran source files prior to
parsing to fix up some constructs (such as continuation lines) that are
problematic for the OpenFortranParser front end. It can also be run as a script,
as seen below.

Example:
    To invoke this script, do: ::
        ./preprocessor.py <infile> <outfile>

where `infile` is the name of the input file, and `outfile` is the name of the
file to which the preprocessed code will be written.

Author:
    Saumya Debray
"""

import sys
from collections import OrderedDict
from typing import List, Dict, Tuple
from delphi.translators.for2py.syntax import (
    line_is_comment,
    line_is_comment_ext,
    line_is_continuation,
    line_is_continued,
    line_is_executable,
    line_is_pgm_unit_end,
    line_is_pgm_unit_separator,
    line_is_pgm_unit_start,
    program_unit_name,
)


def separate_trailing_comments(lines: List[str]) -> List[Tuple[int, str]]:
    """Given a list of numbered Fortran source code lines, i.e., pairs of the
       form (n, code_line) where n is a line number and code_line is a line
       of code, separate_trailing_comments() behaves as follows: for each
       pair (n, code_line) where code_line can be broken into two parts -- a
       code portion code_part and a trailing comment portion comment_part, such
       that code_part and comment_part are both non-empty, it replaces the
       pair (n, code_line) by two pairs (n, comment_part) and (n, code_part).
       The return value is the resulting list of numbered lines.
    """

    i = 0
    while i < len(lines):
        code_line = lines[i]
        if not line_is_comment(code_line):
            (code_part, comment_part) = split_trailing_comment(code_line)
            if comment_part is not None:
                lines[i] = code_part
        i += 1

    return lines


def merge_continued_lines(lines):
    """Given a list of numered Fortran source code lines, i.e., pairs of the
       form (n, code_line) where n is a line number and code_line is a line
       of code, merge_continued_lines() merges sequences of lines that are
       indicated to be continuation lines.
    """

    # Merge continuation lines
    chg = True
    while chg:
        chg = False
        i = 0
        while i < len(lines):
            line = lines[i]
            if line_is_continuation(line):
                assert i > 0
                prev_line_code = lines[i - 1]
                curr_line_code = line.lstrip()[1:]  # remove continuation  char
                merged_code = prev_line_code.rstrip() + \
                              " " + \
                              curr_line_code.lstrip() + \
                              "\n"
                lines[i - 1] = merged_code
                lines.pop(i)
                chg = True
            elif line_is_continued(line):
                assert i < len(lines)-1  # there must be a next line
                next_line_code = lines[i + 1]
                curr_line_code = line.rstrip()[
                    :-1
                ].rstrip()  # remove continuation  char
                merged_code = curr_line_code + " " + next_line_code.lstrip()
                lines[i] = merged_code
                lines.pop(i+1)
                chg = True

            i += 1

    return lines


def discard_comments(lines):
    return [line for line in lines if not line_is_comment(line)]


def split_trailing_comment(line: str) -> str:
    """Takes a line and splits it into two parts (code_part, comment_part)
    where code_part is the line up to but not including any trailing
    comment (the '!' comment character and subsequent characters
    to the end of the line), while comment_part is the trailing comment.
    Args:
        line: A line of Fortran source code.
    Returns:
        A pair (code_part, comment_part) where comment_part is the trailing
        comment.  If the line does not contain any trailing comment, then
        comment_part is None.
    """

    if line.find("!") == -1:
        return (line, None)

    i = 0
    while i < len(line):
        if line[i] == "'":
            j = line.find("'", i + 1)
            if j == -1:
                sys.stderr.write("WEIRD: unbalanced quote ': line = " + line)
                return (line, None)
            else:
                i = j + 1
        elif line[i] == '"':
            j = line.find('"', i + 1)
            if j == -1:
                sys.stderr.write('WEIRD: unbalanced quote ": line = ' + line)
                return (line, None)
            else:
                i = j + 1
        elif line[i] == "!":  # partial-line comment
            comment_part = line[i:]
            code_part = line[:i].rstrip() + "\n"
            return (code_part, comment_part)
        else:
            i += 1

    return (line, None)


def preprocess(lines):
    lines = [line for line in lines if line.rstrip() != ""]
    lines = separate_trailing_comments(lines)
    lines = discard_comments(lines)
    lines = merge_continued_lines(lines)
    return lines

def discard_line(line):
    return (line is None or line.strip() == '')


def process(inputLines: List[str]) -> str:
    """process() provides the interface used by an earlier version of this
       preprocessor."""
    lines = preprocess(inputLines)
    actual_lines = [
        line
        for line in lines
        if not discard_line(line)
    ]
    return "".join(alines)


def preprocess_file(infile, outfile):
    with open(infile, mode="r", encoding="latin-1") as f:
        inputLines = f.readlines()
        lines = preprocess(inputLines)

    with open(outfile, "w") as f:
        for _, line in lines:
            if line is not None:
                f.write(line)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.stderr.write("*** USAGE: preprocessor.py <infile> <outfile>\n")
        sys.exit(1)

    infile, outfile = sys.argv[1], sys.argv[2]
    preprocess_file(infile, outfile)

