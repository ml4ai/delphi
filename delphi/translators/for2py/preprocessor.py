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

import os
import sys
import re
from collections import OrderedDict
from typing import List, Dict, Tuple
from delphi.translators.for2py.syntax import (
    line_is_comment,
    line_is_continuation,
    line_is_continued,
    line_is_include,
)


def separate_trailing_comments(lines: List[str]) -> List[Tuple[int, str]]:
    """Given a list of Fortran source code linesseparate_trailing_comments()
       removes partial-line comments and returns the resulting list of lines.
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


def merge_continued_lines(lines, f_ext):
    """Given a list of Fortran source code lines, merge_continued_lines() 
       merges sequences of lines that are indicated to be continuation lines
       and returns the resulting list of source lines.  The argument f_ext
       gives the file extension of the input file: this determines whether
       we have fixed-form or free-form syntax, which determines how 
       continuation lines are written.  
    """
    chg = True
    while chg:
        chg = False
        i = 0
        while i < len(lines):
            line = lines[i]
            if line_is_continuation(line, f_ext):
                assert i > 0, "Weird continuation line (line {}): {}".format(i+1, line)
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
    return [line for line in lines 
                 if not (line_is_comment(line) or line.strip() == '')]


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
        elif line[i] == "!" and i != 5:  # partial-line comment
            comment_part = line[i:]
            code_part = line[:i].rstrip() + "\n"
            return (code_part, comment_part)
        else:
            i += 1

    return (line, None)


def path_to_target(infile, target):
    # if target is already specified via an absolute path, return that path
    if target[0] == '/':
        return target

    # if infile has a path specified, specify target relative to that path
    pos = infile.rfind('/')
    if pos >= 0:
        path_to_infile = infile[:pos]
        return "{}/{}".format(path_to_infile, target)

    # otherwise simply return target
    return target


def process_includes(lines, infile):
    """ process_includes() processes INCLUDE statements, which behave like
        the #include preprocessor directive in C.
    """
    chg = True
    while chg:
        chg = False
        include_idxs = [i for i in range(len(lines))
                          if line_is_include(lines[i]) is not None]

        # include_idxs is a list of the index positions of INCLUDE statements.
        # Each such statement is processed by replacing it with the contents
        # of the file it mentions.  We process include_idxs in reverse so that
        # processing an INCLUDE statement does not change the index position of 
        # any remaining INCLUDE statements.
        for idx in reversed(include_idxs):
            chg = True
            include_f = line_is_include(lines[idx])
            assert include_f is not None
            include_path = path_to_target(infile, include_f)
            incl_lines = preprocess_file(include_path)
            lines = lines[:idx] + incl_lines + lines[idx+1:]            

    return lines


def refactor_select_case(lines):
    """Search for lines that are CASE statements and refactor their structure
    such that they are always in a i:j form. This means any CASE statement that
    is in the form <:3> will be <Inf:3>. This is done so that the FortranOFP
    recognizes the <:3> and <3:> structures properly.
    """
    prefix_regex = re.compile(r'([(,]):([\d\w+])', re.I)
    suffix_regex = re.compile(r'([\d\w+]):([),])', re.I)
    i = 0
    while i < len(lines):
        code_line = lines[i]
        if prefix_regex.search(code_line):
            match_list = re.findall(prefix_regex, code_line)
            code_line = re.sub(prefix_regex, f"{match_list[0][0]}'-Inf':"
                                             f"{match_list[0][1]}", code_line)
        if suffix_regex.search(code_line):
            match_list = re.findall(suffix_regex, code_line)
            code_line = re.sub(suffix_regex, f"{match_list[0][0]}:'Inf'"
                                             f"{match_list[0][1]}", code_line)

        lines[i] = code_line
        i += 1
    return lines


def preprocess(lines, infile):
    _, f_ext = os.path.splitext(infile)
    lines = [line for line in lines if line.rstrip() != ""]
    lines = separate_trailing_comments(lines)
    lines = discard_comments(lines)
    lines = merge_continued_lines(lines, f_ext)
    lines = process_includes(lines, infile)
    lines = refactor_select_case(lines)
    return lines


def process(inputLines: List[str], infile: str) -> str:
    """process() provides the interface used by an earlier version of this
       preprocessor."""
    lines = preprocess(inputLines, infile)
    return "".join(lines)


def preprocess_file(infile):
    with open(infile, mode="r", encoding="latin-1") as f:
        inputLines = f.readlines()
        lines = preprocess(inputLines, infile)
        return lines


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.stderr.write("*** USAGE: preprocessor.py <infile> <outfile>\n")
        sys.exit(1)

    infile, outfile = sys.argv[1], sys.argv[2]
    lines = preprocess_file(infile)

    with open(outfile, "w") as f:
        for line in lines:
            f.write(line)
