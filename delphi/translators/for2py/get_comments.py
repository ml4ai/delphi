#!/usr/bin/env python
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
    comment dictionary; the comment dictionary maps each of the following
    categories to a list of comment strings:

    -- 'head' : whole-line comments just before the subprogram start;
    -- 'neck' : whole-line comments just after the subprogram start;
    -- 'foot' : whole-line comments just after the subprogram ends; and
    -- 'internal' : comments internal to the function (curently marked
    by "marker statements" of the form i_g_n_o_r_e__m_e___NNN = .True.
    where NNN is the line number of the comment.  Internal comments
    are maintained as a dictionary that maps the variables
    i_g_n_o_r_e__m_e___NNN to lists of comment strings.

    If a subprogram does not have comments for any of these categories, the
    corresponding entry in the comment dictionary is [].

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

import os
import sys
import re
from collections import OrderedDict
from delphi.translators.for2py.syntax import (
    line_is_comment,
    line_starts_subpgm,
    line_ends_subpgm,
    line_is_continuation,
)
import json

INTERNAL_COMMENT_MARKER = r"\s*(i_g_n_o_r_e__m_e___\d+)\s*=\s*\.True\."
RE_INTERNAL_COMMENT_MARKER = re.compile(INTERNAL_COMMENT_MARKER, re.I)

################################################################################
#                                                                              #
#                              COMMENT EXTRACTION                              #
#                                                                              #
################################################################################


def get_comments(src_file_name: str):
    curr_comment = []
    curr_fn, prev_fn, curr_marker = None, None, None
    in_neck = False
    comments = OrderedDict()
    lineno = 1

    comments["$file_head"] = None
    comments["$file_foot"] = None

    _, f_ext = os.path.splitext(src_file_name)

    with open(src_file_name, "r", encoding="latin-1") as f:
        for line in f:
            if line_is_comment(line):
                curr_comment.append(line)
            else:
                if comments["$file_head"] is None:
                    comments["$file_head"] = curr_comment

                f_start, f_name_maybe = line_starts_subpgm(line)
                if f_start:
                    f_name = f_name_maybe

                    prev_fn = curr_fn
                    curr_fn = f_name

                    if prev_fn is not None:
                        comments[prev_fn]["foot"] = curr_comment

                    comments[curr_fn] = init_comment_map(
                        curr_comment, [], [], OrderedDict()
                    )
                    curr_comment = []
                    in_neck = True
                elif line_ends_subpgm(line):
                    curr_comment = []
                elif line_is_continuation(line, f_ext):
                    lineno += 1
                    continue
                else:
                    if in_neck:
                        comments[curr_fn]["neck"] = curr_comment
                        in_neck = False
                        curr_comment = []
                    else:
                        match = re.match(RE_INTERNAL_COMMENT_MARKER, line)
                        if match != None:
                            curr_marker = match.group(1)
                            curr_comment = []
                        elif curr_marker != None:
                            comments[curr_fn]["internal"][
                                curr_marker
                            ] = curr_comment
                            curr_marker = None
                            curr_comment = []

            lineno += 1

    # if there's a comment at the very end of the file, make it the foot
    # comment of curr_fn
    if curr_comment != [] and comments.get(curr_fn):
        comments[curr_fn]["foot"] = curr_comment
        comments["$file_foot"] = curr_comment

    if comments["$file_head"] is None:
        comments["$file_head"] = []
    if comments["$file_foot"] is None:
        comments["$file_foot"] = []

    return comments


def init_comment_map(head_cmt, neck_cmt, foot_cmt, internal_cmt):
    return {
        "head": head_cmt,
        "neck": neck_cmt,
        "foot": foot_cmt,
        "internal": internal_cmt,
    }


def print_comments(comments):

    for fn, comment in comments.items():
        if fn in ("$file_head", "$file_foot"):  # file-level comments
            print(fn + ":")
            for line in comment:
                print(f"    {line.rstrip()}")
            print("")
        else:  # subprogram comments
            print(f"Function: {fn}")
            for ccat in ["head", "neck", "foot"]:
                print(f"  {ccat}:")
                for line in comment[ccat]:
                    print(f"    {line.rstrip()}")
                print("")

            if comment["internal"] != {}:
                print("  internal:")
                for marker in comment["internal"]:
                    comment_line_no = marker[len("i_g_n_o_r_e__m_e___") :]
                    print(f"  line {comment_line_no}:")
                    for line in comment["internal"][marker]:
                        print(f"    {line.rstrip()}")
                    print("")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.stderr.write(f"Usage: {sys.argv[0]} filename\n")
        sys.exit(1)

    filename = sys.argv[1]
    comments = get_comments(filename)
    print_comments(comments)
    clean_filename = filename[filename.rfind("/") + 1 : filename.rfind(".")]
    with open(
        "{}_extracted_comments.json".format(clean_filename.lower()), "w"
    ) as outfile:
        json.dump(comments, outfile)
