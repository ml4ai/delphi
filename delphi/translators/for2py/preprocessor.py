"""
This module implements functions to preprocess Fortran source files prior to
parsing to fix up some constructs (such as continuation lines) that are
problematic for the OpenFortranParser front end.

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
                assert i > 0, "Weird continuation line (line {}): {}".format(
                    i + 1, line
                )
                prev_line_code = lines[i - 1]
                curr_line_code = line.lstrip()[1:]  # remove continuation  char
                merged_code = (
                    prev_line_code.rstrip()
                    + " "
                    + curr_line_code.lstrip()
                    + "\n"
                )
                lines[i - 1] = merged_code
                lines.pop(i)
                chg = True
            elif line_is_continued(line):
                assert i < len(lines) - 1  # there must be a next line
                next_line_code = lines[i + 1]
                curr_line_code = line.rstrip()[
                    :-1
                ].rstrip()  # remove continuation  char
                merged_code = curr_line_code + " " + next_line_code.lstrip()
                lines[i] = merged_code
                lines.pop(i + 1)
                chg = True

            i += 1
    return lines


def discard_comments(lines):
    return [
        line
        for line in lines
        if not (line_is_comment(line) or line.strip() == "")
    ]


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
    if target[0] == "/":
        return target

    # if infile has a path specified, specify target relative to that path
    pos = infile.rfind("/")
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
        include_idxs = [
            i
            for i in range(len(lines))
            if line_is_include(lines[i]) is not None
        ]

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
            incl_lines = get_preprocessed_lines_from_file(include_path)
            lines = lines[:idx] + incl_lines + lines[idx + 1:]

    return lines


def refactor_select_case(lines):
    """Search for lines that are CASE statements and refactor their structure
    such that they are always in a i:j form. This means any CASE statement that
    is in the form <:3> will be <Inf:3>. This is done so that the FortranOFP
    recognizes the <:3> and <3:> structures properly.
    """
    prefix_regex = re.compile(r"([(,])\s*:\s*(-?[\d\w+])", re.I)
    suffix_regex = re.compile(r"(-?[\d\w+])\s*:\s*([),])", re.I)
    i = 0
    while i < len(lines):
        code_line = lines[i]
        if prefix_regex.search(code_line):
            match_list = re.findall(prefix_regex, code_line)
            code_line = re.sub(
                prefix_regex,
                f"{match_list[0][0]}'-Inf':" f"{match_list[0][1]}",
                code_line,
            )
        if suffix_regex.search(code_line):
            match_list = re.findall(suffix_regex, code_line)
            code_line = re.sub(
                suffix_regex,
                f"{match_list[0][0]}:'Inf'" f"{match_list[0][1]}",
                code_line,
            )

        lines[i] = code_line
        i += 1
    return lines


# The regular expressions defined below are used for processing implicit array
# declarations, which the preprocessor converts into explicit array declarations

BASE_TYPES = r"^(\s*)(integer|real|double\s+precision|complex|character" \
             r"|logical)\s+(.*)"
RE_BASE_TYPES = re.compile(BASE_TYPES, re.I)

KWDS = r"\s*(DIMENSION|FUNCTION)\s*.*"
RE_KWDS = re.compile(KWDS, re.I)

IMPLICIT_ARRAY = r"(\w+)\((\w+|\w+,\w+|\w+:\w+)\)"
RE_IMPLICIT_ARRAY = re.compile(IMPLICIT_ARRAY, re.I)

VAR_OR_ARRAY = r"\s*(\w+)(\((\w+|\w+,\w+|\w+:\w+)\))?"
RE_VAR_OR_ARRAY = re.compile(VAR_OR_ARRAY, re.I)

DECL_CONTINUATION = r"\s*,\s*"
RE_DECL_CONTINUATION = re.compile(DECL_CONTINUATION, re.I)


def implicit_array_decl_parameters(line):
    """ If line contains an implicit array declaration, extract and return
        the following parameters: the initial indentation, the type of the
        array, and the rest of the line after the type; otherwise return None
    """
    match = RE_BASE_TYPES.match(line)
    if match is None:
        return None

    indentation = match.group(1)
    type = match.group(2)
    rest = match.group(3)

    if type.lower() == "character":
        match = re.match(r"\s*(\(\s*len\s*=\s*\d+\s*\)|\*\s*\d+)", rest)
        if match is not None:
            char_parms = match.group(1)
            type += char_parms
            rest = rest[match.end():]

    # If the the rest of the string begins with specific keywords
    # like DIMENSION or FUNCTION, this is not an implicit declaration.
    match = RE_KWDS.match(rest)
    if match is not None:
        return None

    # If the line does not match the pattern for an implicit array,
    # it does not have an implicit array declaration
    match = RE_IMPLICIT_ARRAY.search(rest)
    if match is None:
        return None

    return indentation, type, rest


def fix_implicit_array_decls(lines):
    out_lines = []
    for line in lines:
        implicit_decl_parms = implicit_array_decl_parameters(line)
        if implicit_decl_parms is None:
            out_lines.append(line)
            continue
        else:
            (indentation, type, rest) = implicit_decl_parms
            decls = {}
            arr_name = arr_size = None
            match2 = RE_VAR_OR_ARRAY.match(rest)
            while match2 is not None:
                arr_name, arr_size = match2.group(1), match2.group(2)
                if arr_size is not None:
                    arr_size = arr_size[1:-1]
                else:
                    arr_size = 0
    
                if arr_size in decls:
                    decls[arr_size] += ", " + arr_name
                else:
                    decls[arr_size] = arr_name
    
                # get the rest of the string if appropriate
                n = match2.end()
                if n < len(rest):
                    rest = rest[n:]
                else:
                    rest = ""
    
                # process any comma separator if present
                match3 = RE_DECL_CONTINUATION.match(rest)
                if match3 is not None:
                    n = match3.end()
                    rest = rest[n:]
    
                match2 = RE_VAR_OR_ARRAY.match(rest)
    
            # finally, construct the output lines with implicit declarations
            # replaced by explicit ones
            new_lines = []
            for sz in decls:
                if sz != 0:
                    new_lines.append("{}{}, DIMENSION({}) :: {}\n".format(
                        indentation, type, sz, decls[sz]))
                else:
                    new_lines.append("{}{} :: {}\n".format(indentation, type,
                                                           decls[sz]))
    
            out_lines.extend(new_lines)

    return out_lines


def preprocess_lines(lines, infile, forModLogGen=False):
    _, f_ext = os.path.splitext(infile)
    lines = [line for line in lines if line.rstrip() != ""]
    lines = separate_trailing_comments(lines)
    lines = discard_comments(lines)
    lines = merge_continued_lines(lines, f_ext)
    lines = fix_implicit_array_decls(lines)
    # For module log file generation, we do not need to
    # preprocess any included external files, so skip in
    # such case.
    if not forModLogGen:
        lines = process_includes(lines, infile)
    lines = refactor_select_case(lines)
    return [x.lower() for x in lines]


def get_preprocessed_lines_from_file(infile, forModLogGen=False):
    with open(infile, mode="r", encoding="latin-1") as f:
        lines = f.readlines()
    return preprocess_lines(lines, infile, forModLogGen)


