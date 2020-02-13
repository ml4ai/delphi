#!/usr/bin/env python3.7

""" This file contains code to carry out a simple estimate of the amount of
    Fortran code 'handled' by for2py.

    COMMAND-LINE INVOCATION:

        python3.7 measure-coverage.py <directory with Fortran code files>


    ADDING HANDLED LANGUAGE CONSTRUCTS:
    As the set of language features handled by for2py grows, they should be
    incorporated into this script.  This can be done as follows:

    1) Write a regular expression to recognize that feature (see examples
       under "SYNTAX MATCHING" below).

    2) Add the regular expression to the list for the variable HANDLED.
"""

import os
import sys
import preprocessor
from syntax import *

FORTRAN_EXTENSIONS = ['.f', '.f90', '.for']

################################################################################
#                                                                              #
#                                SYNTAX MATCHING                               #
#                                                                              #
################################################################################

# Regular expressions that specify patterns for various Fortran constructs.
# These are very similar to the constructs in the file syntax.py, but only
# include constructs that are currently handled in for2py.

FN_START = r"\s*(\w*\s*){0,2}function\s+(\w+)\s*\("
RE_FN_START = re.compile(FN_START, re.I)

PGM_UNIT = r"\s*\w*\s*(program|module|subroutine|(\w*\s*){0,2}function)\s+(\w+)"
RE_PGM_UNIT_START = re.compile(PGM_UNIT, re.I)

PGM_UNIT_SEP = r"\s+contains(\W+)"
RE_PGM_UNIT_SEP = re.compile(PGM_UNIT_SEP, re.I)

PGM_UNIT_END = r"\s*[a-z]*\s*end\s+(program|module|subroutine|function)\s+"
RE_PGM_UNIT_END = re.compile(PGM_UNIT_END, re.I)

SUBPGM_END = r"\s*end\s+"
RE_SUBPGM_END = re.compile(SUBPGM_END, re.I)

ASSG_STMT = r"\s*(\d+|&)?\s*.*=\s*"
RE_ASSG_STMT = re.compile(ASSG_STMT, re.I)

IMPLICIT_STMT = r"\s*implicit\s+"
RE_IMPLICIT_STMT = re.compile(IMPLICIT_STMT, re.I)

CALL_STMT = r"\s*(\d+|&)?\s*call\s*"
RE_CALL_STMT = re.compile(CALL_STMT, re.I)

IO_STMT = r"\s*(\d+|&)?\s*(open|close|read|write|print|format|rewind)\W*"
RE_IO_STMT = re.compile(IO_STMT, re.I)

DO_STMT = r"\s*(\d+|&)?\s*do\s*"
RE_DO_STMT = re.compile(DO_STMT, re.I)

ENDDO_STMT = r"\s*(\d+|&)?\s*end\s*do\s*"
RE_ENDDO_STMT = re.compile(ENDDO_STMT, re.I)

ENDIF_STMT = r"\s*(\d+|&)?\s*end\s*if\s*"
RE_ENDIF_STMT = re.compile(ENDIF_STMT, re.I)

GOTO_STMT = r"\s*(\d+|&)?\s*go\s*to\s*"
RE_GOTO_STMT = re.compile(GOTO_STMT, re.I)

IF_STMT = r"\s*(\d+|&)?\s*(if|elseif|else)\s*"
RE_IF_STMT = re.compile(IF_STMT, re.I)

INCLUDE_STMT_1 = r"\s*(\d+)?\s*include\s+'(\w+(\.\w*)?)'"
RE_INCLUDE_STMT_1 = re.compile(INCLUDE_STMT_1, re.I)
INCLUDE_STMT_2 = r'\s*(\d+)?\s*include\s+"(\w+(\.\w*)?)"'
RE_INCLUDE_STMT_2 = re.compile(INCLUDE_STMT_2, re.I)

PAUSE_STMT = r"\s*(\d+|&)?\s*pause\s*"
RE_PAUSE_STMT = re.compile(PAUSE_STMT, re.I)

USE_STMT = r"\s*(\d+|&)?\s*use\s*"
RE_USE_STMT = re.compile(USE_STMT, re.I)

RETURN_STMT = r"\s*(\d+|&)?\s*return\s*"
RE_RETURN_STMT = re.compile(RETURN_STMT, re.I)

CYCLE_STMT = r"\s*(\d+|&)?\s*cycle\s*"
RE_CYCLE_STMT = re.compile(CYCLE_STMT, re.I)

EXIT_STMT = r"\s*(\d+|&)?\s*exit\s*"
RE_EXIT_STMT = re.compile(EXIT_STMT, re.I)

SAVE_STMT = r"\s*(\d+|&)?\s*save\s*"
RE_SAVE_STMT = re.compile(SAVE_STMT, re.I)

SELECT_STMT = r"\s*(\d+|&)?\s*select\s*case\s*"
RE_SELECT_STMT = re.compile(SELECT_STMT, re.I)

ENDSELECT_STMT = r"\s*(\d+|&)?\s*end\s*select\s*"
RE_ENDSELECT_STMT = re.compile(ENDSELECT_STMT, re.I)

CASE_STMT = r"\s*(\d+|&)?\s*case\s*"
RE_CASE_STMT = re.compile(CASE_STMT, re.I)

STOP_STMT = r"\s*(\d+|&)?\s*stop\s*"
RE_STOP_STMT = re.compile(STOP_STMT, re.I)

TYPE_NAMES = r"^\s*(integer|real|double\s+precision|logical|dimension|type" \
             r"|character)\W*"
RE_TYPE_NAMES = re.compile(TYPE_NAMES, re.I)

HANDLED = [
    RE_FN_START,
    RE_PGM_UNIT_START,
    RE_PGM_UNIT_SEP,
    RE_PGM_UNIT_END,
    RE_SUBPGM_END,
    RE_ASSG_STMT,
    RE_CALL_STMT,
    RE_CYCLE_STMT,
    RE_EXIT_STMT,
    RE_IMPLICIT_STMT,
    RE_INCLUDE_STMT_1,
    RE_INCLUDE_STMT_2,
    RE_IO_STMT,
    RE_DO_STMT,
    RE_ENDDO_STMT,
    RE_ENDIF_STMT,
    RE_GOTO_STMT,
    RE_IF_STMT,
    RE_PAUSE_STMT,
    RE_RETURN_STMT,
    RE_SAVE_STMT,
    RE_STOP_STMT,
    RE_TYPE_NAMES,
    RE_USE_STMT,
]

KEYWD = r"\s*(\d+|&)?\s*([a-z]+).*"
RE_KEYWD = re.compile(KEYWD)


def line_is_handled(fname, line):
    unhandled_keywds, unhandled_lines = set(), set()

    match = RE_KEYWD.match(line.strip().lower())
    if match is not None:
        first_wd = match.group(2)
    else:
        first_wd = None

    for handled_construct in HANDLED:
        if handled_construct.match(line) is not None:
            return True, unhandled_keywds, unhandled_lines

    if first_wd in F_KEYWDS:
        unhandled_keywds.add(first_wd)
    else:
        unhandled_lines.add((fname, line))

    return False, unhandled_keywds, unhandled_lines


################################################################################
#                                                                              #
#                                FILE PROCESSING                               #
#                                                                              #
################################################################################

def get_code_lines(fname):
    print("@@@ FILE: " + fname)
    return preprocessor.preprocess_file(fname)


def process_lines(fname, lines):
    handled_files = 1
    unhandled_keywds = {}
    unhandled_lines = set()
    nlines = len(lines)
    nhandled = 0
    for line in lines:
        handled, u_keywds, u_lines = line_is_handled(fname, line)
        if handled:
            nhandled += 1
        else:
            handled_files = 0
            unhandled_lines |= u_lines
            for keywd in u_keywds:
                if keywd in unhandled_keywds:
                    unhandled_keywds[keywd] += 1
                else:
                    unhandled_keywds[keywd] = 1

    return nlines, nhandled, unhandled_keywds, unhandled_lines, handled_files


def process_file(fname):
    code_lines = get_code_lines(fname)
    results = process_lines(fname, code_lines)
    return results


def process_dir(dirname):
    unhandled_keywds, unhandled_lines = {}, set()
    nfiles, ntot, nhandled, handled_files = 0, 0, 0, 0

    abs_path = os.path.abspath(dirname)
    print(f"processing: {dirname}")
    
    list_of_files = os.listdir(dirname)
    for fname in list_of_files:
        full_path_to_file = abs_path + "/" + fname
        if os.path.isdir(full_path_to_file):
            nf1, nt1, nh1, uk1, ul1, hf = process_dir(full_path_to_file)
            nfiles += nf1
            ntot += nt1
            nhandled += nh1
            handled_files += hf
            for keywd in uk1:
                if keywd in unhandled_keywds:
                    unhandled_keywds[keywd] += uk1[keywd]
                else:
                    unhandled_keywds[keywd] = uk1[keywd]

            unhandled_lines |= ul1
        else:
            _, fext = os.path.splitext(fname)
            if fext in FORTRAN_EXTENSIONS: 
                ftot, fhandled, u_keywds, u_lines, h_files = process_file(
                    full_path_to_file)
                ntot += ftot
                nhandled += fhandled
                handled_files += h_files

                for keywd in u_keywds:
                    if keywd in unhandled_keywds:
                        unhandled_keywds[keywd] += u_keywds[keywd]
                    else:
                        unhandled_keywds[keywd] = u_keywds[keywd]
                unhandled_lines |= u_lines
                nfiles += 1
            else:
                sys.stderr.write(f"    *** Ignoring {fname} [unrecognized "
                                 f"extension]\n")

    return (nfiles, ntot, nhandled, unhandled_keywds, unhandled_lines,
            handled_files)


def usage():
    sys.stderr.write("Usage: measure-coverage.py <src-directory>\n")


def errmsg(msg):
    sys.stderr.write(msg + "\n")
    sys.exit(1)


def print_results(results):
    nfiles, ntot, nhandled, u_keywds, u_lines, h_files = results
    pct_handled = h_files/nfiles * 100
    print(f"FILES: total = {nfiles}; handled = {h_files} "
          f"[{pct_handled:.1f}%]\n")

    pct_handled = nhandled/ntot * 100
    print(f"LINES: total = {ntot}; handled = {nhandled} [{pct_handled:.1f}%]\n")

    if u_keywds != set():
        print("UNHANDLED KEYWORDS:")
        print("------------------")
        for item in u_keywds:
            print(f"    {item} : {u_keywds[item]}")

    if u_lines != set():
        print("UNHANDLED LINES:")
        print("---------------")
        for item in u_lines:
            print(f"    {item}")


def main():
    if len(sys.argv) < 2:
        usage()
        sys.exit(1)

    results = process_dir(sys.argv[1])
    print_results(results)


if __name__ == "__main__":
    main()
