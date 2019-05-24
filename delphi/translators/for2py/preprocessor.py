#!/usr/bin/env python3

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
    line_is_continuation,
    line_is_executable,
    line_is_pgm_unit_end,
    line_is_pgm_unit_start,
    program_unit_name,
)


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

    enumerated_lines = list(enumerate(lines, 1))
    i = 0
    while i < len(enumerated_lines):
        (n, code_line) = enumerated_lines[i]
        if not line_is_comment(code_line):
            (code_part, comment_part) = split_trailing_comment(code_line)
            if comment_part is not None:
                enumerated_lines[i] = (n, comment_part)
                enumerated_lines.insert(i + 1, (n, code_part))
        i += 1

    return enumerated_lines


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
    while chg:
        chg = False
        i = 0
        while i < len(lines) - 1:
            ln0, ln1 = lines[i], lines[i + 1]
            if line_is_comment(ln0[1]) and line_is_continuation(ln1[1]):
                # swap the code portions of lines[i] and lines[i+1]
                lines[i], lines[i + 1] = (ln0[0], ln1[1]), (ln1[0], ln0[1])
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
                merged_code = prev_line_code.rstrip() + curr_line_code.lstrip()

                lines[i - 1] = (prev_linenum, merged_code)
                lines.pop(i)
                chg = True
            i += 1

    return lines


# We use a finite-state machine to keep track of where the line currently
# being processed sits w.r.t. the structure of the code; this affects how
# comments should be handled.  The FSM has the following set of states:
#
#     { "outside", "in_neck", "in_body" }.
#
# Here, "outside" refers to program points outside any program unit such as
# programs, subprograms, or modules; "in_neck" refers to the portion of
# code within a subprogram between the subprogram header and the first line
# of executable code; and "in_body" refers to the portion of the code between
# the first line of executable code and the end of the subprogram.
#
# State transitions for the FSM are given by the dictionary TRANSITIONS;
# here "comment", "pgm_unit_start", "exec_stmt", etc., relate to the nature of
# the line being processed.

TRANSITIONS = {
    "outside": {"comment": "outside", "pgm_unit_start": "in_neck"},
    "in_neck": {
        "comment": "in_neck",
        "exec_stmt": "in_body",
        "other": "in_neck",
    },
    "in_body": {
        "comment": "in_body",
        "exec_stmt": "in_body",
        "pgm_unit_end": "outside",
    },
}


def type_of_line(line):
    """Given a line of code, type_of_line() returns a string indicating
       what kind of code it is."""

    if line_is_comment(line):
        return "comment"
    elif line_is_executable(line):
        return "exec_stmt"
    elif line_is_pgm_unit_end(line):
        return "pgm_unit_end"
    else:
        if line_is_pgm_unit_start(line):
            return "pgm_unit_start"
        else:
            return "other"


def extract_comments(
    lines: List[Tuple[int, str]]
) -> Tuple[List[Tuple[int, str]], Dict[str, List[str]]]:
    """Given a list of numbered lines from a Fortran file where comments
       internal to subprogram bodies have been moved out into their own lines,
       extract_comments() extracts comments into a dictionary and replaces
       each comment internal to subprogram bodies with a marker statement.
       It returns a pair (code, comments) where code is a list of numbered
       lines with comments removed and marker statements (plus corresponding
       variable declarations) added; and comments is a dictionary mapping
       marker statement variables to the corresponding comments."""

    curr_comment = []
    curr_fn, prev_fn, curr_marker = None, None, None
    comments = OrderedDict()

    # curr_state refers to the state of the finite-state machine (see above)
    curr_state = "outside"

    comments["$file_head"] = []
    comments["$file_foot"] = []

    for i in range(len(lines)):
        (linenum, line) = lines[i]

        # determine what kind of line this is
        line_type = type_of_line(line)

        # process the line appropriately
        if curr_state == "outside":
            assert line_type in ("comment", "pgm_unit_start"), (
                line_type,
                line,
            )
            if line_type == "comment":
                curr_comment.append(line)
                lines[i] = (linenum, None)
            else:
                # line_type == "pgm_unit_start"
                pgm_unit_name = program_unit_name(line)
                comments["$file_head"] = curr_comment

                if prev_fn is not None:
                    comments[prev_fn]["foot"] = curr_comment

                prev_fn = curr_fn
                curr_fn = pgm_unit_name

                internal_comments = OrderedDict()

                comments[curr_fn] = init_comment_map(
                    curr_comment, [], [], internal_comments
                )
                curr_comment = []

        elif curr_state == "in_neck":
            assert line_type in ("comment", "exec_stmt", "other")
            if line_type == "comment":
                curr_comment.append(line)
                lines[i] = (linenum, None)
            elif line_type == "exec_stmt":
                comments[curr_fn]["neck"] = curr_comment
                curr_comment = []
            else:
                pass  # nothing to do -- continue

        elif curr_state == "in_body":
            assert line_type in (
                "comment",
                "exec_stmt",
                "pgm_unit_end",
            ), f"[Line {linenum}]: {line}"

            if line_type == "comment":
                # Ignore empty lines, which are technically comments but which
                # don't contribute any semantic content.
                if line != "\n":
                    marker_var = f"{INTERNAL_COMMENT_PREFIX}_{linenum}"
                    marker_stmt = f"        {marker_var} = .True.\n"
                    internal_comments[marker_var] = line
                    lines[i] = (linenum, marker_stmt)
            else:
                pass  # nothing to do -- continue

        # update the current state
        curr_state = TRANSITIONS[curr_state][line_type]

    # if there's a comment at the very end of the file, make it the foot
    # comment of curr_fn
    if curr_comment != [] and comments.get(curr_fn):
        comments[curr_fn]["foot"] = curr_comment
        comments["$file_foot"] = curr_comment

    return (lines, comments)


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
            comment_part = line[i:]
            code_part = line[:i].rstrip() + "\n"
            return (code_part, comment_part)
        else:
            i += 1

    return (line, None)


def process(inputLines: List[str]) -> str:
    """process() provides the interface used by an earlier version of this
       preprocessor."""
    lines = separate_trailing_comments(inputLines)
    merge_continued_lines(lines)
    (lines, comments) = extract_comments(lines)
    actual_lines = [
        line[1]
        for line in lines
        if line[1] is not None and "i_g_n_o_r_e___m_e_" not in line[1]
    ]
    return "".join(actual_lines)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.stderr.write("*** USAGE: preprocessor.py <infile> <outfile>\n")
        sys.exit(1)

    infile, outfile = sys.argv[1], sys.argv[2]
    with open(infile, mode="r", encoding="latin-1") as f:
        inputLines = f.readlines()

    lines = separate_trailing_comments(inputLines)
    merge_continued_lines(lines)

    (lines, comments) = extract_comments(lines)

    with open(outfile, "w") as f:
        for _, line in lines:
            if line is not None:
                f.write(line)
