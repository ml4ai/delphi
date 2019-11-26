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


# IGNORE_INTERNAL_COMMENTS: if set to True, internal comments are dropped.
IGNORE_INTERNAL_COMMENTS = True

# INTERNAL_COMMENT_PREFIX is a prefix used for marker variables associated
# with comments internal to subprogram bodies.
INTERNAL_COMMENT_PREFIX = "i_g_n_o_r_e___m_e_"


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
        (n, code_line) = lines[i]
        if not line_is_comment(code_line):
            (code_part, comment_part) = split_trailing_comment(code_line)
            if comment_part is not None:
                lines[i] = (n, comment_part)
                lines.insert(i + 1, (n, code_part))
        i += 1

    return lines


def merge_continued_lines(lines):
    """Given a list of numered Fortran source code lines, i.e., pairs of the
       form (n, code_line) where n is a line number and code_line is a line
       of code, merge_continued_lines() merges sequences of lines that are
       indicated to be continuation lines.
    """

    # Before a continuation line L1 is merged with the line L0 before it (and
    # presumably the one L1 is continuing), ensure that L0 is not a comment.
    # If L0 is a comment, swap L0 and L1.
    chg = True
    swaps = set()
    while chg:
        chg = False
        i = 0
        while i < len(lines) - 1:
            ln0, ln1 = lines[i], lines[i + 1]
            if (line_is_comment_ext(ln0[1]) and line_is_continuation(ln1[1])) \
               or (line_is_continued(ln0[1]) and line_is_comment_ext(ln1[1])):
                if (i, i+1) not in swaps:
                    # swap the code portions of lines[i] and lines[i+1]
                    lines[i], lines[i + 1] = (ln0[0], ln1[1]), (ln1[0], ln0[1])
                    swaps.add((i,i+1))  # to prevent infinite loops
                else:
                   # If we get here, there is a pair of adjacent lines that
                   # are about to go into an infinite swap sequence; one of them
                   # must be a comment.  We delete the comment.
                   if line_is_comment_ext(ln0[1]):
                       lines.pop(i)
                   else:
                       assert line_is_comment_ext(ln1[1])
                       lines.pop(i+1)
                chg = True

            i += 1

    # Merge continuation lines
    chg = True
    while chg:
        chg = False
        i = 0
        while i < len(lines):
            line = lines[i]
            if line_is_continuation(line[1]):
                assert i > 0
                (prev_linenum, prev_line_code) = lines[i - 1]
                curr_line_code = line[1].lstrip()[
                    1:
                ]  # remove continuation  char
                merged_code = prev_line_code.rstrip() + \
                              " " + \
                              curr_line_code.lstrip() + \
                              "\n"
                lines[i - 1] = (prev_linenum, merged_code)
                lines.pop(i)
                chg = True
            elif line_is_continued(line[1]):
                assert i < len(lines)-1  # there must be a next line
                (next_linenum, next_line_code) = lines[i + 1]
                curr_line_code = line[1].rstrip()[
                    :-1
                ].rstrip()  # remove continuation  char
                merged_code = curr_line_code + " " + next_line_code.lstrip()
                lines[i] = (i, merged_code)
                lines.pop(i+1)
                chg = True

            i += 1

    return lines


def merge_adjacent_comment_lines(lines):
    """Given a list of numered Fortran source code lines, i.e., pairs of the
       form (n, code_line) where n is a line number and code_line is a line
       of code, merge_adjacent_comment_lines() merges sequences of lines that are
       indicated to be comment lines.
    """

    i = 0
    while i < len(lines)-1:
        lnum, line = lines[i]
        if line_is_comment(line):
            j = i+1
            while j < len(lines) and line_is_comment(lines[j][1]):
                line += lines[j][1]
                lines.pop(j)
                # pop() removes a line so lines[j] now refers to the next line

            lines[i] = (lnum, line)
        i += 1

    return lines


def type_of_line(line):
    """Given a line of code, type_of_line() returns a string indicating
       what kind of code it is."""

    if line.strip() == "":
        return "empty"

    if line_is_comment(line):
        return "comment"
    elif line_is_executable(line):
        return "exec_stmt"
    elif line_is_pgm_unit_end(line):
        return "pgm_unit_end"
    elif line_is_pgm_unit_start(line):
        return "pgm_unit_start"
    elif line_is_pgm_unit_separator(line):
        return "pgm_unit_sep"
    else:
        return "other"


def discard_comments(lines: List[Tuple[int, str]]) -> List[Tuple[int, str]]:

    curr_fn, prev_fn, curr_marker = None, None, None

    # curr_state refers to the state of the finite-state machine (see above)
    curr_state = "outside"

    for i in range(len(lines)):
        (linenum, line) = lines[i]

        # determine what kind of line this is
        line_type = type_of_line(line)

        # process the line appropriately
        if line_type == "comment":
            lines[i] = (linenum, None)

    return lines


def init_comment_map(head_cmt, neck_cmt, foot_cmt, internal_cmt):
    return {
        "head": head_cmt,
        "neck": neck_cmt,
        "foot": foot_cmt,
        "internal": internal_cmt,
    }


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
    enum_lines = list(enumerate(lines, 1))

    # Discard empty lines. While these are technically comments, they provide
    # no semantic content.  
    enum_lines = [line for line in enum_lines if line[1].rstrip() != ""]

    enum_lines = separate_trailing_comments(enum_lines)
    enum_lines = merge_continued_lines(enum_lines)
    enum_lines = merge_adjacent_comment_lines(enum_lines)
    return discard_comments(enum_lines)


def discard_line(line):
    return (line is None or 
            line.strip() == '' or
            INTERNAL_COMMENT_PREFIX in line)


def process(inputLines: List[str]) -> str:
    """process() provides the interface used by an earlier version of this
       preprocessor."""
    lines = preprocess(inputLines)
    actual_lines = [
        line[1]
        for line in lines
        if not discard_line(line[1])
    ]
    return "".join(actual_lines)


def preprocess_file(infile, outfile):
    with open(infile, mode="r", encoding="latin-1") as f:
        inputLines = f.readlines()
        lines, comments = preprocess(inputLines)

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

