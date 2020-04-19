""" Various utility routines for processing Fortran syntax.  """

import re
from typing import Tuple, Optional

################################################################################
#                                                                              #
#                                   COMMENTS                                   #
#                                                                              #
################################################################################


def line_is_comment(line: str) -> bool:
    """From FORTRAN Language Reference
    (https://docs.oracle.com/cd/E19957-01/805-4939/z40007332024/index.html)

    A line with a c, C, '*', d, D, or ! in column one is a comment line, except
    that if the -xld option is set, then the lines starting with D or d are
    compiled as debug lines. The d, D, and ! are nonstandard.

    If you put an exclamation mark (!) in any column of the statement field,
    except within character literals, then everything after the ! on that
    line is a comment.

    A totally blank line is a comment line (we ignore this here).

    Args:
        line

    Returns:
        True iff line is a comment, False otherwise.

    """

    return line[0] in "cCdD*!"


################################################################################
#                                                                              #
#                              CONTINUATION LINES                              #
#                                                                              #
################################################################################

FIXED_FORM_EXT = ('.f', '.for', '.blk', '.inc', '.gin')


def line_is_continuation(line: str, f_ext: str) -> bool:
    """ From FORTRAN 77  Language Reference
        (https://docs.oracle.com/cd/E19957-01/805-4939/6j4m0vn6l/index.html)

    A statement takes one or more lines; the first line is called the initial
    line; the subsequent lines are called the continuation lines.  You can
    format a source line in either of two ways: Standard fixed format,
    or Tab format.

    In Standard Fixed Format, continuation lines are identified by a nonblank, 
    nonzero in column 6.

    Tab-Format source lines are defined as follows: A tab in any of columns 
    1 through 6, or an ampersand in column 1, establishes the line as a 
    tab-format source line.  If the tab is the first nonblank character, the 
    text following the tab is scanned as if it started in column 7.
    Continuation lines are identified by an ampersand (&) in column 1, or a 
    nonzero digit after the first tab.    

    Args:
        line
    Returns:
        True iff line is a continuation line, else False.  Currently this
        is used only for fixed-form input files, i.e., f_ext in ('.f', '.for')
    """

    if line_is_comment(line):
        return False

    if f_ext in FIXED_FORM_EXT:
        if line[0] == '\t':
            return line[1] in "123456789"
        else:
            return len(line) > 5 and not (line[5] == ' ' or line[5] == '0')
    if line[0] == '&':
        return True

    return False


def line_is_continued(line: str) -> bool:
    """
    Args:
        line
    Returns:
        True iff line is continued on the next line.  This is a Fortran-90
        feature and indicated by a '&' at the end of the line.
    """

    if line_is_comment(line):
        return False

    llstr = line.rstrip()
    return len(llstr) > 0 and llstr[-1] == "&"


################################################################################
#                                                                              #
#                           FORTRAN LINE PROCESSING                            #
#                                                                              #
################################################################################

# Regular expressions that specify patterns for various Fortran constructs.

SUB_START = r"\s*subroutine\s+(\w+)\s*\("
RE_SUB_START = re.compile(SUB_START, re.I)

# The start of a function can look like any of the following:
#
#       function myfunc(...
#       integer function myfunc(...
#       double precision myfunc(...

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

INCLUDE_STMT_1 = r"\s*(\d+)?\s*include\s+'(\w+(\.\w*)?)'"
RE_INCLUDE_STMT_1 = re.compile(INCLUDE_STMT_1, re.I)
INCLUDE_STMT_2 = r'\s*(\d+)?\s*include\s+"(\w+(\.\w*)?)"'
RE_INCLUDE_STMT_2 = re.compile(INCLUDE_STMT_2, re.I)

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

PAUSE_STMT = r"\s*(\d+|&)?\s*pause\s*"
RE_PAUSE_STMT = re.compile(PAUSE_STMT, re.I)

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

TYPE_NAMES = r"^\s*(integer|real|double\s+precision|complex|character|logical" \
             r"|dimension|type|parameter)\W*"
RE_TYPE_NAMES = re.compile(TYPE_NAMES, re.I)

SUB_DEF = r"\s*subroutine\s+(?P<subroutine>(?P<name>.*?)\((?P<args>.*)\))"
RE_SUB_DEF = re.compile(SUB_DEF, re.I)

END_SUB = r"\s*end\s+subroutine\s+"
END_MODULE = r"\s*end\s+module\s+"

PGM_START = r"\s*(?P<pgm>(interface|module))\s+(?P<name>(\w*))"
RE_PGM_START = re.compile(PGM_START, re.I)

PGM_END = r"\s*[a-z]*\s*end\s+(?P<pgm>(module|subroutine|function|interface|type))\s*(?P<name>(\w*))"
RE_PGM_END = re.compile(PGM_END, re.I)

VARIABLE_DEC = r"\s*(?P<type>(double|float|int|integer|logical|real|str|string)),?\s(::)*(?P<variables>(.*))"
RE_VARIABLE_DEC = re.compile(VARIABLE_DEC, re.I)

DERIVED_TYPE_DEF = r"\s*type\s*(?P<type>(?P<name>(\w*)))"
RE_DERIVED_TYPE_DEF = re.compile(DERIVED_TYPE_DEF, re.I)

DERIVED_TYPE_VAR = r"\s*type\s*\((?P<type>.*)\)\s(?P<variables>(.*))"
RE_DERIVED_TYPE_VAR = re.compile(DERIVED_TYPE_VAR, re.I)

CHARACTER = r"\s*(character\*(\(\*\)|\d*))\s*(?P<variables>.*)"
RE_CHARACTER = re.compile(CHARACTER, re.I)

IMP_ARRAY = r"(?P<var>\w*)\s*\("
RE_IMP_ARRAY = re.compile(IMP_ARRAY, re.I)

CLASS_DEF = r"(?P<class>class)\s*(?P<name>(\w*)):"
RE_CLASS_DEF = re.compile(CLASS_DEF, re.I)

# EXECUTABLE_CODE_START is a list of regular expressions matching
# lines that can start the executable code in a program or subroutine.
# It is used to for handling comments internal to subroutine bodies.
EXECUTABLE_CODE_START = [
    RE_ASSG_STMT,
    RE_CALL_STMT,
    RE_CASE_STMT,
    RE_CYCLE_STMT,
    RE_DO_STMT,
    RE_ENDDO_STMT,
    RE_ENDIF_STMT,
    RE_ENDSELECT_STMT,
    RE_EXIT_STMT,
    RE_GOTO_STMT,
    RE_IF_STMT,
    RE_IO_STMT,
    RE_PAUSE_STMT,
    RE_RETURN_STMT,
    RE_SELECT_STMT,
    RE_STOP_STMT
]

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
    if match is not None:
        f_name = match.group(1)
        return True, f_name

    match = RE_FN_START.match(line)
    if match is not None:
        f_name = match.group(2)
        return True, f_name

    return False, None


def line_is_pgm_unit_start(line):
    match = RE_PGM_UNIT_START.match(line)
    return match is not None


def line_is_pgm_unit_end(line):
    match = RE_PGM_UNIT_END.match(line)
    return match is not None


def line_is_pgm_unit_separator(line):
    match = RE_PGM_UNIT_SEP.match(line)
    return match is not None


def program_unit_name(line: str) -> str:
    """
     Given a line that starts a program unit, i.e., a program, module,
      subprogram, or function, this function returns the name associated
      with that program unit.
    """
    match = RE_PGM_UNIT_START.match(line)
    assert match is not None
    return match.group(2)


def line_ends_subpgm(line: str) -> bool:
    """
    Args:
        line
    Returns:
        True if line is the last line of a subprogram definition, else False.
    """
    match = RE_SUBPGM_END.match(line)
    return match is not None


def line_is_executable(line: str) -> bool:
    """line_is_executable() returns True iff the line can start an
       executable statement in a program."""

    if line_is_comment(line):
        return False

    if re.match(RE_TYPE_NAMES, line):
        return False

    for exp in EXECUTABLE_CODE_START:
        if re.match(exp, line) is not None:
            return True

    return False
            

def line_is_include(line: str) -> str:
    """ line_is_include() : if the argument is an INCLUDE statement, returns
        the argument to the INCLUDE statement (a file name); otherwise
        returns None.
    """
    match = RE_INCLUDE_STMT_1.match(line)
    if match is not None:
        return match.group(2)

    match = RE_INCLUDE_STMT_2.match(line)
    if match is not None:
        return match.group(2)

    return None


def line_has_implicit_array(line: str) -> Tuple[bool, Optional[str]]:
    """
    This function checks for the implicit array declaration (i.e.
    real myArray(10)) without DIMENSION keyword.

    Args:
        line
    Returns:
        (True, var) if the line holds implicit array declaration;
        (False, None) if the line does not hold implicit array declaration
    """

    match = RE_IMP_ARRAY.match(line)
    if match:
        var = match['var']
        return True, var
    return False, None


def has_subroutine(lines: str) -> bool:
    """
    This function searches end of subroutine syntax
    from the passed string of lines to find out whether
    any subroutines are presented in lines or not.

    Args:
        lines
    Returns:
        (True) if subroutine end syntax presented in lines.
        (False) if subroutine end syntax does not present in lines.
    """
    return re.search(END_SUB, lines)


def has_module(lines: str) -> bool:
    """This function searches end of module syntax
    from the passed string of lines to find out whether
    any modules are presented in lines or not.

    Args:
        lines
    Returns:
        (True) if module end syntax presented in lines.
        (False) if module end syntax does not present in lines.
    """
    return re.search(END_MODULE, lines)


def line_is_func_start(line: str) -> bool:
    """
    This function checks whether the line indicates a function entry or not.

    Args:
        line.
    Returns:
        (True) if the line is beginning of function definition.
        (False) if the line is not a beginning of function dedfinition.
    """
    match = RE_FN_START.match(line)
    return match is not None


def variable_declaration(line: str) -> Tuple[bool, Optional[list]]:
    """
    Indicates whether a line in the program is a variable declaration or not.

    Args:
        line
    Returns:
       (True, v_type, list) if line has one of variable declaration syntax;
       (False, None, None) if line does not have any variable declaration syntax
    """

    match_var = RE_VARIABLE_DEC.match(line)
    if match_var:
        v_type = match_var['type']
        variable_list = match_var['variables']
        return (True, v_type, variable_list)
    
    match_dtype = RE_DERIVED_TYPE_VAR.match(line)
    if match_dtype:
        v_type = match_dtype['type']
        variable_list = match_dtype['variables']
        return (True, v_type, variable_list)

    match_char = RE_CHARACTER.match(line)
    if match_char:
        variable_list = match_char['variables']
        return (True, "character", variable_list)

    return (False, None, None)


def subroutine_definition(line: str) -> Tuple[bool, Optional[list]]:
    """
    Indicates whether a line in the program is the first line of a subroutine
    definition and extracts subroutine name and the arguments.

    Args:
        line
    Returns:
       (True, list) if line begins a definition for subroutine;
       (False, None) if line does not begin a subroutine definition.
    """
    match = RE_SUB_DEF.match(line)
    if match:
        f_name = match['name'].strip()
        f_args = match['args'].replace(' ', '').split(',')
        return True, [f_name, f_args]

    return False, None


def pgm_end(line: str) -> Tuple[bool, str, str]:
    """
    Indicates whether a line is the last line of a program.

    Args:
        line
    Returns:
       (True, pgm, name) if line is the last line of a definition
       either for interface or module;
       (False, None, None) if line is not the last line of either
       interface or module.
    """
    match = RE_PGM_END.match(line)
    if match:
        pgm = match['pgm']
        pgm_name = match['name']
        return True, pgm, pgm_name
    return (False, None, None)


def line_starts_pgm(line: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Indicates whether a line is the first line of either interface or module.

    Args:
        line
    Returns:
       (True, pgm, name) if line begins a definition either for interface or
       module;
       (False, None, None) if line is not a beginning of either interface or
       module.
    """

    match = RE_PGM_START.match(line)
    if match:
        pgm = match['pgm'].strip()
        name = match['name'].strip()
        return True, pgm, name
    return False, None, None


def is_class_def(line: str) -> Tuple[bool, Optional[str]]:
    """
    Indicates whether a line in the translated Python program is the
    first line of a class definition.
    Args:
        linee
    Return:
        True if line is a class definition;
        False if line is not a class definition.
    """
    match = RE_CLASS_DEF.match(line)
    if match:
        return True, match['name']
    else:
        return False, None

################################################################################
#                                                                              #
#                       FORTRAN KEYWORDS AND INTRINSICS                        #
#                                                                              #
################################################################################

# F_KEYWDS : Fortran keywords
#      SOURCE: Keywords in Fortran Wiki 
#              (http://fortranwiki.org/fortran/show/Keywords)
#
#              tutorialspoint.com
#              (https://www.tutorialspoint.com/fortran/fortran_quick_guide.htm) 
#
# NOTE1: Because of the way this code does tokenization, it currently does
#       not handle keywords that consist of multiple space-separated words
#       (e.g., 'do concurrent', 'sync all').
#
# NOTE2: This list of keywords is incomplete.

F_KEYWDS = frozenset(['abstract', 'allocatable', 'allocate', 'assign',
                      'assignment', 'associate', 'asynchronous', 'backspace',
                      'bind', 'block', 'block data', 'call', 'case',
                      'character', 'class', 'close', 'codimension', 'common',
                      'complex', 'contains', 'contiguous', 'continue',
                      'critical', 'cycle', 'data', 'deallocate', 'default',
                      'deferred', 'dimension', 'do', 'double precision',
                      'else', 'elseif', 'elsewhere', 'end', 'endif',
                      'endfile', 'endif', 'entry', 'enum', 'enumerator',
                      'equivalence', 'error', 'exit', 'extends', 'external',
                      'final', 'flush', 'forall', 'format', 'function',
                      'generic', 'goto', 'if', 'implicit', 'import', 'in',
                      'include', 'inout', 'inquire', 'integer', 'intent',
                      'interface', 'intrinsic', 'kind', 'len', 'lock',
                      'logical', 'module', 'namelist', 'non_overridable',
                      'nopass', 'nullify', 'only', 'open', 'operator',
                      'optional', 'out', 'parameter', 'pass', 'pause',
                      'pointer', 'print', 'private', 'procedure', 'program',
                      'protected', 'public', 'pure', 'read', 'real',
                      'recursive', 'result', 'return', 'rewind', 'rewrite',
                      'save', 'select', 'sequence', 'stop', 'submodule',
                      'subroutine', 'sync', 'target', 'then', 'type',
                      'unlock', 'use', 'value', 'volatile', 'wait', 'where',
                      'while', 'write'])

# F_INTRINSICS : intrinsic functions
#     SOURCE: GNU gfortran manual: 
#     https://gcc.gnu.org/onlinedocs/gfortran/Intrinsic-Procedures.html

F_INTRINSICS = frozenset(['abs', 'abort', 'access', 'achar', 'acos', 'acosd',
                          'acosh', 'adjustl', 'adjustr', 'aimag', 'aint',
                          'alarm', 'all', 'allocated', 'alog', 'alog10', 
                          'amax0', 'amax1', 'amin0', 'amin1', 'and',
                          'anint', 'any','asin', 'asind', 'asinh', 'associated',
                          'atan', 'atand', 'atan2', 'atan2d', 'atanh', 'atomic_add',
                          'atomic_and', 'atomic_cas', 'atomic_define',
                          'atomic_fetch_add', 'atomic_fetch_and',
                          'atomic_fetch_or', 'atomic_fetch_xor', 'atomic_or',
                          'atomic_ref', 'atomic_xor', 'backtrace',
                          'bessel_j0', 'bessel_j1', 'bessel_jn', 'bessel_y0',
                          'bessel_y1', 'bessel_yn', 'bge', 'bgt', 'bit_size',
                          'ble', 'blt', 'btest', 'c_associated',
                          'c_f_pointer', 'c_f_procpointer', 'c_funloc',
                          'c_loc', 'c_sizeof', 'ceiling', 'char', 'chdir',
                          'chmod', 'cmplx', 'co_broadcast', 'co_max',
                          'co_min', 'co_reduce', 'co_sum',
                          'command_argument_count', 'compiler_options',
                          'compiler_version', 'complex', 'conjg', 'cos',
                          'cosd', 'cosh', 'cotan', 'cotand', 'count',
                          'cpu_time', 'cshift', 'ctime', 'date_and_time',
                          'dble', 'dcmplx', 'digits', 'dim', 'dot_product',
                          'dprod', 'dreal', 'dshiftl', 'dshiftr', 'dtime',
                          'eoshift', 'epsilon', 'erf', 'erfc', 'erfc_scaled',
                          'etime', 'event_query', 'execute_command_line',
                          'exit', 'exp', 'exponent', 'extends_type_of',
                          'fdate', 'fget', 'fgetc', 'floor', 'flush', 'fnum',
                          'fput', 'fputc', 'fraction', 'free', 'fseek',
                          'fstat', 'ftell', 'gamma', 'gerror', 'getarg',
                          'get_command', 'get_command_argument', 'getcwd',
                          'getenv', 'get_environment_variable', 'getgid',
                          'getlog', 'getpid', 'getuid', 'gmtime', 'hostnm',
                          'huge', 'hypot', 'iachar', 'iall', 'iand', 'iany',
                          'iargc', 'ibclr', 'ibits', 'ibset', 'ichar',
                          'idate', 'ieor', 'ierrno', 'image_index', 'index',
                          'int', 'int2', 'int8', 'ior', 'iparity', 'irand',
                          'is_iostat_end', 'is_iostat_eor', 'isatty',
                          'ishft', 'ishftc', 'isnan', 'itime', 'kill',
                          'kind', 'lbound', 'lcobound', 'leadz', 'len',
                          'len_trim', 'lge', 'lgt', 'link', 'lle', 'llt',
                          'lnblnk', 'loc', 'log', 'log10', 'log_gamma',
                          'logical', 'long', 'lshift', 'lstat', 'ltime',
                          'malloc', 'maskl', 'maskr', 'matmul', 'max',
                          'maxexponent', 'maxloc', 'maxval', 'mclock',
                          'mclock8', 'merge', 'merge_bits', 'min',
                          'minexponent', 'minloc', 'minval', 'mod', 'modulo',
                          'move_alloc', 'mvbits', 'nearest', 'new_line',
                          'nint', 'norm2', 'not', 'null', 'num_images', 'or',
                          'pack', 'parity', 'perror', 'popcnt', 'poppar',
                          'precision', 'present', 'product', 'radix', 'ran',
                          'rand', 'random_number', 'random_seed', 'range',
                          'rank ', 'real', 'rename', 'repeat', 'reshape',
                          'rrspacing', 'rshift', 'same_type_as', 'scale',
                          'scan', 'secnds', 'second', 'selected_char_kind',
                          'selected_int_kind', 'selected_real_kind',
                          'set_exponent', 'shape', 'shifta', 'shiftl',
                          'shiftr', 'sign', 'signal', 'sin', 'sind', 'sinh',
                          'size', 'sizeof', 'sleep', 'spacing', 'spread',
                          'sqrt', 'srand', 'stat', 'storage_size', 'sum',
                          'symlnk', 'system', 'system_clock', 'tan', 'tand',
                          'tanh', 'this_image', 'time', 'time8', 'tiny',
                          'trailz', 'transfer', 'transpose', 'trim',
                          'ttynam', 'ubound', 'ucobound', 'umask', 'unlink',
                          'unpack', 'verify', 'xor', 'amax1', 'amin1'])


################################################################################
#                                                                              #
#                       NEGATED OPERATIONS MAPPING                             #
#                                                                              #
################################################################################

NEGATED_OP = {
                ".le.": ".gt.",
                ".ge.": ".lt.",
                ".lt.": ".ge.",
                ".gt.": ".le.",
                ".eq.": ".ne.",
                ".ne.": ".eq.",
                "<=": ">",
                ">=": "<",
                "==": "!=",
                "<": ">=",
                ">": "<=",
                "!=": "==",
              }
