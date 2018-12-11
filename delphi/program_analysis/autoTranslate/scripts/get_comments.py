#!/usr/bin/env python3

"""
Purpose:
    Read the Fortran source file specified and return subprogram
    names together with the associated subprogram-level comments.

Author:
    Saumya Debray

Example:
    Command-line invocation:::

        ./get_comments.py <src_file_name>

    Programmatic invocation:::

        comments = get_comments(src_file_name)

    The returned value is a dictionary that maps each subprogram name to a
    comment dictionary; the comment dictionary maps each of the categories
    'head', 'neck', and 'foot' to a list of comment strings.  If a
    particular subprogram does not have comments for any of these
    categories, the corresponding entries in the comment dictionary are [].
"""

import sys, re
from collections import *
from delphi.program_analysis.autoTranslate.scripts.fortran_syntax import *
from typing import Tuple, Optional

DEBUG = False

################################################################################
#                                                                              #
#                              COMMENT EXTRACTION                              #
#                                                                              #
################################################################################

# This code considers only subprogram-level comments, i.e., whole-line comments
# that come:
#     (a) immediately before the subprogram begins (comment type: "head"); or
#     (b) immediately after the subprogram begins (comment type: "neck"); or
#     (c) immediately after the subprogram ends (comment type: "foot").
#
# A subprogram-level comment that occurs between two subprograms F and G will
# be considered to be F's foot-comment and G's head-comment.
#
# The comments extracted from a file are maintained as a dictionary that maps
# each subprogram name to a comment dictionary; the comment dictionary maps
# each of the categories 'head', 'neck' 'foot' to a list of comment strings.
# If a particular subprogram does not have comments for any of these categories,
# that category is mapped to [] by the comment dictionary for that subprogram.


def get_comments(src_file_name: str):
    curr_comment = []
    curr_fn, prev_fn = None, None
    in_neck = False
    collect_comments = True
    comments = OrderedDict()
    lineno = 1

    with open(src_file_name, "r", encoding="latin-1") as f:
        for line in f:
            if line_is_comment(line) and collect_comments:
                curr_comment.append(line)
            else:
                f_start, f_name = line_starts_subpgm(line)
                if f_start:
                    if DEBUG:
                        print(f"<<< START: line {lineno}, fn = {f_name}")

                    if prev_fn != None:
                        comments[prev_fn]["foot"] = curr_comment

                    prev_fn = curr_fn
                    curr_fn = f_name

                    comments[curr_fn] = init_comment_map(curr_comment, [], [])
                    curr_comment = []
                    in_neck = True
                elif line_ends_subpgm(line):
                    if DEBUG:
                        print(f">>> END: line {lineno}, fn = {f_name}")

                    curr_comment = []
                    collect_comments = True
                elif line_is_continuation(line):
                    continue
                else:
                    if in_neck:
                        comments[curr_fn]["neck"] = curr_comment
                        in_neck = False
                    collect_comments = False

            lineno += 1

    # if there's a comment at the very end of the file, make it the foot
    # comment of curr_fn
    if curr_comment != []:
        comments[curr_fn]["foot"] = curr_comment
    return comments


def init_comment_map(head_cmt, neck_cmt, foot_cmt):
    return {"head": head_cmt, "neck": neck_cmt, "foot": foot_cmt}


def print_comments(comments):

    for fn in comments:
        print(fn)
        print(f"Function: {fn}")
        fn_comment = comments[fn]

        for ccat in ["head", "neck", "foot"]:
            print(f"  {ccat}:")
            for line in fn_comment[ccat]:
                print(f"    {line.rstrip()}")
            print("")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.stderr.write(f"Usage: {sys.argv[0]} filename\n")
        sys.exit(1)

    comments = get_comments(sys.argv[1])
    print_comments(comments)
