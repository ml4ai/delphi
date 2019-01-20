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

    In addition to the subprogram-level comments mentioned above, the returned
    dictionary also has entries for two "file-level" comments:

        -- any comment at the beginning of the file (before the first function)
           can be accessed using the key "$file_head" (this comment is also
           the head-comment for the first subprogram in the file); and
        -- any comment at the end of the file (after the last function)
           can be accessed using the key "$file_foot" (this comment is also
           the foot-comment for the last subprogram in the file).

    If either the file-head or the file-foot comment is missing, the
    corresponding entries in the comment dictionary are [].
"""

import sys, re
from collections import *
from delphi.translators.for2py.scripts.fortran_syntax import *
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
#
# In addition to the subprogram-level comments mentioned above, the comment
# dictionary also has entries for two "file-level" comments: 
#
#    -- any comment at the beginning of the file (before the first function)
#       can be accessed using the key "$file_head" (this comment is also
#       the head-comment for the first subprogram in the file); and
#    -- any comment at the end of the file (after the last function)
#       can be accessed using the key "$file_foot" (this comment is also
#       the foot-comment for the last subprogram in the file).
#
# If either the file-head or the file-foot comment is missing, the 
# corresponding entries in the comment dictionary are [].


def get_comments(src_file_name: str):
    curr_comment = []
    curr_fn, prev_fn = None, None
    in_neck = False
    collect_comments = True
    comments = OrderedDict()
    lineno = 1

    comments["$file_head"] = []
    comments["$file_foot"] = []

    with open(src_file_name, "r", encoding="latin-1") as f:
        for line in f:
            if line_is_comment(line) and collect_comments:
                curr_comment.append(line)
            else:
                f_start, f_name_maybe = line_starts_subpgm(line)
                if f_start:
                    f_name = f_name_maybe

                    if DEBUG:
                        print(f"<<< START: line {lineno}, fn = {f_name}; prev_fn = {str(prev_fn)}")

                    if curr_fn == None:
                        comments["$file_head"] = curr_comment

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
                    lineno += 1
                    continue
                else:
                    if in_neck:
                        comments[curr_fn]["neck"] = curr_comment
                        in_neck = False
                    collect_comments = False

            lineno += 1

    # if there's a comment at the very end of the file, make it the foot
    # comment of curr_fn
    if curr_comment != [] and comments.get(curr_fn):
        comments[curr_fn]["foot"] = curr_comment
        comments["$file_foot"] = curr_comment
    return comments


def init_comment_map(head_cmt, neck_cmt, foot_cmt):
    return {"head": head_cmt, "neck": neck_cmt, "foot": foot_cmt}


def print_comments(comments):

    for fn in comments:
        fn_comment = comments[fn]

        if fn == "$file_head" or fn == "$file_foot":    # file-level comments
            print(fn+":")
            for line in fn_comment:
                print(f"    {line.rstrip()}")
            print("")
        else:                                           # subprogram comments
            print(f"Function: {fn}")
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
