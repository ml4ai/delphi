#!/usr/bin/env python3

"""
    File: get_comments.py
    Author: Saumya Debray
    Purpose: Read the Fortran source file specified and return subprogram
            names together with the associated subprogram-level comments.
    Usage:
        Command-line invocation:

            get_comments.py src_file_name

        Programmatic invocation:

            comments = get_comments(src_file_name)

        The returned value is a dictionary that maps each subprogram name to
        a comment dictionary; the comment dictionary maps each of the categories
        'head', 'neck' 'foot' to a list of comment strings.  If a particular
        subprogram does not have comments for any of these categories, the
        corresponding entries in the comment dictionary are [].
"""

import sys, re
from collections import *
from typing import Tuple, Optional

DEBUG = False

################################################################################
#                                                                              #
#                                   COMMENTS                                   #
#                                                                              #
################################################################################

def line_is_comment(line: str) -> bool:
    """
    From FORTRAN Language Reference
    (https://docs.oracle.com/cd/E19957-01/805-4939/z40007332024/index.html):

    A line with a c, C, *, d, D, or ! in column one is a comment line, except
    that if the -xld option is set, then the lines starting with D or d are
    compiled as debug lines. The d, D, and ! are nonstandard.

    If you put an exclamation mark (!) in any column of the statement field,
    except within character literals, then everything after the ! on that
    line is a comment.

    A totally blank line is a comment line.

    Args:
        line

    Returns:
        True iff line is a comment, False otherwise.

    """

    if line[0] in "cCdD*!":
        return True

    llstr = line.strip()
    if len(llstr) == 0 or llstr[0] == "!":
        return True

    return False


################################################################################
#                                                                              #
#                           FORTRAN LINE PROCESSING                            #
#                                                                              #
################################################################################

# SUB_START, FN_START, and SUBPGM_END are regular expressions that specify
# patterns for the Fortran syntax for the start of subroutines and functions,
# and their ends, respectively.  The corresponding re objects are RE_SUB_START,
# and RE_FN_START, and RE_SUBPGM_END.

SUB_START = "\s*subroutine\s+(\w+)\s*\("
RE_SUB_START = re.compile(SUB_START, re.I)

FN_START = "\s*function\s+(\w+)\s*\("
RE_FN_START = re.compile(FN_START, re.I)

SUBPGM_END = "\s*end\s+"
RE_SUBPGM_END = re.compile(SUBPGM_END, re.I)



def line_starts_subpgm(line: str) -> Tuple[bool, Optional[str]]:
    """
    Indicates whether a line in the program is the first line of a subprogram
    definition.

    Args:
        line
    Returns:
       (True, f_name) if line begins a definition for subprogram f_name;
       (False, None) if line does not begin a subprogram definition.
    """

    match = RE_SUB_START.match(line)
    if match != None:
        f_name = match.group(1)
        return (True, f_name)

    match = RE_FN_START.match(line)
    if match != None:
        f_name = match.group(1)
        return (True, f_name)

    return (False, None)


# line_is_continuation(line) 


def line_is_continuation(line: str) -> bool:
    """
    Args:
        line
    Returns:
        True iff line is a continuation line, else False.
    """

    llstr = line.lstrip()
    return len(llstr) > 0 and llstr[0] == "&"



def line_ends_subpgm(line: str) -> bool:
    """
    Args:
        line
    Returns:
        True if line is the last line of a subprogram definition, else False.
    """
    match = RE_SUBPGM_END.match(line)
    return match != None


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
    try:
        src_file = open(src_file_name, mode="r", encoding="latin-1")
    except IOError:
        sys.stderr.write(
            "ERROR: Could not open file {}\n".format(src_file_name)
        )
        sys.exit(1)
    except UnicodeDecodeError:
        sys.stderr.write(
            "ERROR: unicode decoding problems: {}\n".format(src_file)
        )
        sys.exit(1)

    comments = OrderedDict()

    curr_comment = []
    curr_fn, prev_fn = None, None
    in_neck = False
    collect_comments = True

    lineno = 1
    for line in src_file:
        if line_is_comment(line) and collect_comments:
            curr_comment.append(line)
        else:
            f_start, f_name = line_starts_subpgm(line)
            if f_start:
                if DEBUG:
                    print(
                        "<<< START: line {:d}, fn = {}".format(lineno, f_name)
                    )

                if prev_fn != None:
                    comments[prev_fn]["foot"] = curr_comment

                prev_fn = curr_fn
                curr_fn = f_name

                comments[curr_fn] = init_comment_map(curr_comment, [], [])
                curr_comment = []
                in_neck = True
            elif line_ends_subpgm(line):
                if DEBUG:
                    print(">>> END: line {:d}, fn = {}".format(lineno, f_name))

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
        print("Function: {}".format(fn))
        fn_comment = comments[fn]

        for ccat in ["head", "neck", "foot"]:
            print("  {}:".format(ccat))
            for line in fn_comment[ccat]:
                print("    {}".format(line.rstrip()))
            print("")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: {} filename\n".format(sys.argv[0]))
        sys.exit(1)

    comments = get_comments(sys.argv[1])
    print_comments(comments)
