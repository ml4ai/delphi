""" Various utility routines for processing Fortran syntax.  """

import re
from typing import Tuple, Optional

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

# Regular expressions that specify patterns for various Fortran constructs.

SUB_START = r"\s*subroutine\s+(\w+)\s*\("
RE_SUB_START = re.compile(SUB_START, re.I)

FN_START = r"\s*[a-z]*\s*function\s+(\w+)\s*\("
RE_FN_START = re.compile(FN_START, re.I)

PGM_UNIT_START = r"\s+[a-z]*\s+(program|module|subroutine|function)\s+(\w+)"
RE_PGM_UNIT_START = re.compile(PGM_UNIT_START, re.I)

PGM_UNIT_END = r"\s+[a-z]*\s+end\s+(program|module|subroutine|function)\s+"
RE_PGM_UNIT_END = re.compile(PGM_UNIT_END, re.I)

SUBPGM_END = r"\s*end\s+"
RE_SUBPGM_END = re.compile(SUBPGM_END, re.I)

ASSG_STMT = r"\s*(\d+|&)?\s*.*=\s*"
RE_ASSG_STMT = re.compile(ASSG_STMT, re.I)

CALL_STMT = r"\s*(\d+|&)?\s*call\s*"
RE_CALL_STMT = re.compile(CALL_STMT, re.I)

IO_STMT = r"\s*(\d+|&)?\s*(open|close|read|write|print|format)\W*"
RE_IO_STMT = re.compile(IO_STMT, re.I)

DO_STMT = r"\s*(\d+|&)?\s*do\s*"
RE_DO_STMT = re.compile(DO_STMT, re.I)

ENDDO_STMT = r"\s*(\d+|&)?\s*end\s*do\s*"
RE_ENDDO_STMT = re.compile(ENDDO_STMT, re.I)

ENDIF_STMT = r"\s*(\d+|&)?\s*end\s*if\s*"
RE_ENDIF_STMT = re.compile(ENDIF_STMT, re.I)

GOTO_STMT = r"\s*(\d+|&)?\s*goto\s*"
RE_GOTO_STMT = re.compile(GOTO_STMT, re.I)

IF_STMT = r"\s*(\d+|&)?\s*(if|elseif|else)\s*"
RE_IF_STMT = re.compile(IF_STMT, re.I)

PAUSE_STMT = r"\s*(\d+|&)?\s*pause\s*"
RE_PAUSE_STMT = re.compile(PAUSE_STMT, re.I)

RETURN_STMT = r"\s*(\d+|&)?\s*return\s*"
RE_RETURN_STMT = re.compile(RETURN_STMT, re.I)

SAVE_STMT = r"\s*(\d+|&)?\s*save\s*"
RE_SAVE_STMT = re.compile(SAVE_STMT, re.I)

STOP_STMT = r"\s*(\d+|&)?\s*stop\s*"
RE_STOP_STMT = re.compile(STOP_STMT, re.I)

TYPE_NAMES = r"^\s*(integer|real|double|complex|character|logical|dimension|type)\W*"
RE_TYPE_NAMES = re.compile(TYPE_NAMES, re.I)

# EXECUTABLE_CODE_START is a list of regular expressions matching
# lines that can start the executable code in a program or subroutine.
# It is used to for handling comments internal to subroutine bodies.
EXECUTABLE_CODE_START = [
    RE_ASSG_STMT,
    RE_CALL_STMT,
    RE_DO_STMT,
    RE_ENDDO_STMT,
    RE_ENDIF_STMT,
    RE_GOTO_STMT,
    RE_IF_STMT,
    RE_IO_STMT,
    RE_PAUSE_STMT,
    RE_RETURN_STMT,
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
    if match != None:
        f_name = match.group(1)
        return (True, f_name)

    match = RE_FN_START.match(line)
    if match != None:
        f_name = match.group(1)
        return (True, f_name)

    return (False, None)


def line_is_pgm_unit_start(line):
    match = RE_PGM_UNIT_START.match(line)
    return match != None


def line_is_pgm_unit_end(line):
    match = RE_PGM_UNIT_END.match(line)
    return match != None


def program_unit_name(line:str) -> str:
   """Given a line that starts a program unit, i.e., a program, module,
      subprogram, or function, this function returns the name associated
      with that program unit."""
   match = RE_PGM_UNIT_START.match(line)
   assert match != None
   return match.group(2)

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


def line_is_executable(line: str) -> bool:
    """line_is_executable() returns True iff the line can start an
       executable statement in a program."""

    if line_is_comment(line):
        return False

    if re.match(RE_TYPE_NAMES, line):
        return False

    for exp in EXECUTABLE_CODE_START:
        if re.match(exp, line) != None:
            return True

    return False
            


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
    'assignment', 'associate', 'asynchronous', 'backspace', 'bind', 'block',
    'block data', 'call', 'case', 'character', 'class', 'close', 'codimension',
    'common', 'complex', 'contains', 'contiguous', 'continue', 'critical',
    'cycle', 'data', 'deallocate', 'default', 'deferred', 'dimension', 'do',
    'double precision', 'else', 'elseif', 'elsewhere', 'end', 'endif', 'endfile', 
    'endif', 'entry', 'enum', 'enumerator', 'equivalence', 'error', 'exit', 
    'extends', 'external', 'final', 'flush', 'forall', 'format', 'function', 
    'generic', 'goto', 'if', 'implicit', 'import', 'in', 'include', 'inout', 
    'inquire', 'integer', 'intent', 'interface', 'intrinsic', 'kind', 'len', 
    'lock', 'logical', 'module', 'namelist', 'non_overridable', 'nopass', 'nullify',
    'only', 'open', 'operator', 'optional', 'out', 'parameter', 'pass', 'pause',
    'pointer', 'print', 'private', 'procedure', 'program', 'protected', 'public',
    'pure', 'read', 'real', 'recursive', 'result', 'return', 'rewind', 'rewrite',
    'save', 'select', 'sequence', 'stop', 'submodule', 'subroutine', 'sync',
    'target', 'then', 'type', 'unlock', 'use', 'value', 'volatile', 'wait',
    'where', 'while', 'write'])

# F_INTRINSICS : intrinsic functions
#     SOURCE: GNU gfortran manual: 
#     https://gcc.gnu.org/onlinedocs/gfortran/Intrinsic-Procedures.html

F_INTRINSICS = frozenset(['abs', 'abort', 'access', 'achar', 'acos', 'acosd', 
    'acosh', 'adjustl', 'adjustr', 'aimag', 'aint', 'alarm', 'all', 'allocated', 
    'and', 'anint', 'any', 'asin', 'asind', 'asinh', 'associated', 'atan', 
    'atand', 'atan2', 'atan2d', 'atanh', 'atomic_add', 'atomic_and', 'atomic_cas', 
    'atomic_define', 'atomic_fetch_add', 'atomic_fetch_and', 'atomic_fetch_or', 
    'atomic_fetch_xor', 'atomic_or', 'atomic_ref', 'atomic_xor', 'backtrace', 
    'bessel_j0', 'bessel_j1', 'bessel_jn', 'bessel_y0', 'bessel_y1', 'bessel_yn', 
    'bge', 'bgt', 'bit_size', 'ble', 'blt', 'btest', 'c_associated', 
    'c_f_pointer', 'c_f_procpointer', 'c_funloc', 'c_loc', 'c_sizeof', 'ceiling', 
    'char', 'chdir', 'chmod', 'cmplx', 'co_broadcast', 'co_max', 'co_min', 
    'co_reduce', 'co_sum', 'command_argument_count', 'compiler_options', 
    'compiler_version', 'complex', 'conjg', 'cos', 'cosd', 'cosh', 'cotan', 
    'cotand', 'count', 'cpu_time', 'cshift', 'ctime', 'date_and_time', 'dble', 
    'dcmplx', 'digits', 'dim', 'dot_product', 'dprod', 'dreal', 'dshiftl', 
    'dshiftr', 'dtime', 'eoshift', 'epsilon', 'erf', 'erfc', 'erfc_scaled', 
    'etime', 'event_query', 'execute_command_line', 'exit', 'exp', 'exponent', 
    'extends_type_of', 'fdate', 'fget', 'fgetc', 'floor', 'flush', 'fnum', 'fput', 
    'fputc', 'fraction', 'free', 'fseek', 'fstat', 'ftell', 'gamma', 'gerror', 
    'getarg', 'get_command', 'get_command_argument', 'getcwd', 'getenv', 
    'get_environment_variable', 'getgid', 'getlog', 'getpid', 'getuid', 'gmtime', 
    'hostnm', 'huge', 'hypot', 'iachar', 'iall', 'iand', 'iany', 'iargc', 'ibclr', 
    'ibits', 'ibset', 'ichar', 'idate', 'ieor', 'ierrno', 'image_index', 'index', 
    'int', 'int2', 'int8', 'ior', 'iparity', 'irand', 'is_iostat_end', 
    'is_iostat_eor', 'isatty', 'ishft', 'ishftc', 'isnan', 'itime', 'kill', 
    'kind', 'lbound', 'lcobound', 'leadz', 'len', 'len_trim', 'lge', 'lgt', 'link', 
    'lle', 'llt', 'lnblnk', 'loc', 'log', 'log10', 'log_gamma', 'logical', 'long', 
    'lshift', 'lstat', 'ltime', 'malloc', 'maskl', 'maskr', 'matmul', 'max', 
    'maxexponent', 'maxloc', 'maxval', 'mclock', 'mclock8', 'merge', 'merge_bits', 
    'min', 'minexponent', 'minloc', 'minval', 'mod', 'modulo', 'move_alloc', 
    'mvbits', 'nearest', 'new_line', 'nint', 'norm2', 'not', 'null', 'num_images', 
    'or', 'pack', 'parity', 'perror', 'popcnt', 'poppar', 'precision', 'present', 
    'product', 'radix', 'ran', 'rand', 'random_number', 'random_seed', 'range', 
    'rank ', 'real', 'rename', 'repeat', 'reshape', 'rrspacing', 'rshift', 
    'same_type_as', 'scale', 'scan', 'secnds', 'second', 'selected_char_kind', 
    'selected_int_kind', 'selected_real_kind', 'set_exponent', 'shape', 'shifta', 
    'shiftl', 'shiftr', 'sign', 'signal', 'sin', 'sind', 'sinh', 'size', 'sizeof', 
    'sleep', 'spacing', 'spread', 'sqrt', 'srand', 'stat', 'storage_size', 'sum', 
    'symlnk', 'system', 'system_clock', 'tan', 'tand', 'tanh', 'this_image', 
    'time', 'time8', 'tiny', 'trailz', 'transfer', 'transpose', 'trim', 'ttynam', 
    'ubound', 'ucobound', 'umask', 'unlink', 'unpack', 'verify', 'xor'])

