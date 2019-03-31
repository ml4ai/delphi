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

# SUB_START, FN_START, and SUBPGM_END are regular expressions that specify
# patterns for the Fortran syntax for the start of subroutines and functions,
# and their ends, respectively.  The corresponding re objects are RE_SUB_START,
# and RE_FN_START, and RE_SUBPGM_END.

SUB_START = r"\s*subroutine\s+(\w+)\s*\("
RE_SUB_START = re.compile(SUB_START, re.I)

FN_START = r"\s*[a-z]*\s*function\s+(\w+)\s*\("
RE_FN_START = re.compile(FN_START, re.I)

SUBPGM_END = r"\s*end\s+"
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
#
# Each entry in the dictionary F_INTRINSICS is of the form
#
#      fortran_fn : python_tgt
#
# where fortran_fn is the Fortran function; and python_tgt is the corresponding
# Python target, specified as follows:
#
#     -- if the target corresponding to fortran_fn is known, then python_tgt
#        is a tuple (py_fn, fn_type, py_mod), where:
#            -- py_fn is a Python function or operator;
#            -- fn_type is one of: 'FUNC', 'INFIXOP'; and
#            -- py_mod is the module the Python function should be imported from,
#               None if no explicit import is necessary.
#    -- if the target corresponding to fortran_fn is unknown or unavailable,
#       python_tgt is None.

F_INTRINSICS = {
    'abs' : ('abs', 'FUNC', None),
    'abort' : None,
    'access' : None,
    'achar' : None,
    'acos' : ('acos', 'FUNC', 'math'),
    'acosd' : None,
    'acosh' : ('acosh', 'FUNC', 'math'),
    'adjustl' : None,
    'adjustr' : None,
    'aimag' : None,
    'aint' : None,
    'alarm' : None,
    'all' : None,
    'allocated' : None,
    'and' : None,
    'anint' : None,
    'any' : None,
    'asin' : ('asin', 'FUNC', 'math'),
    'asind' : None,
    'asinh' : ('asinh', 'FUNC', 'math'),
    'associated' : None,
    'atan' : ('atan', 'FUNC', 'math'),
    'atand' : None,
    'atan2' : None,
    'atan2d' : None,
    'atanh' : ('atanh', 'FUNC', 'math'),
    'atomic_add' : None,
    'atomic_and' : None,
    'atomic_cas' : None,
    'atomic_define' : None,
    'atomic_fetch_add' : None,
    'atomic_fetch_and' : None,
    'atomic_fetch_or' : None,
    'atomic_fetch_xor' : None,
    'atomic_or' : None,
    'atomic_ref' : None,
    'atomic_xor' : None,
    'backtrace' : None,
    'bessel_j0' : None,
    'bessel_j1' : None,
    'bessel_jn' : None,
    'bessel_y0' : None,
    'bessel_y1' : None,
    'bessel_yn' : None,
    'bge' : None,
    'bgt' : None,
    'bit_size' : None,
    'ble' : None,
    'blt' : None,
    'btest' : None,
    'c_associated' : None,
    'c_f_pointer' : None,
    'c_f_procpointer' : None,
    'c_funloc' : None,
    'c_loc' : None,
    'c_sizeof' : None,
    'ceiling' : ('ceil', 'FUNC', 'math'),
    'char' : None,
    'chdir' : None,
    'chmod' : None,
    'cmplx' : None,
    'co_broadcast' : None,
    'co_max' : None,
    'co_min' : None,
    'co_reduce' : None,
    'co_sum' : None,
    'command_argument_count' : None,
    'compiler_options' : None,
    'compiler_version' : None,
    'complex' : None,
    'conjg' : None,
    'cos' : ('cos', 'FUNC', 'math'),
    'cosd' : None,
    'cosh' : ('cosh', 'FUNC', 'math'),
    'cotan' : None,
    'cotand' : None,
    'count' : None,
    'cpu_time' : None,
    'cshift' : None,
    'ctime' : None,
    'date_and_time' : None,
    'dble' : None,
    'dcmplx' : None,
    'digits' : None,
    'dim' : None,
    'dot_product' : None,
    'dprod' : None,
    'dreal' : None,
    'dshiftl' : None,
    'dshiftr' : None,
    'dtime' : None,
    'eoshift' : None,
    'epsilon' : None,
    'erf' : ('erf', 'FUNC', 'math'),
    'erfc' : ('erfc', 'FUNC', 'math'),
    'erfc_scaled' : None,
    'etime' : None,
    'event_query' : None,
    'execute_command_line' : None,
    'exit' : None,
    'exp' : ('exp', 'FUNC', 'math'),
    'exponent' : None,
    'extends_type_of' : None,
    'fdate' : None,
    'fget' : None,
    'fgetc' : None,
    'floor' : ('floor', 'FUNC', 'math'),
    'flush' : None,
    'fnum' : None,
    'fput' : None,
    'fputc' : None,
    'fraction' : None,
    'free' : None,
    'fseek' : None,
    'fstat' : None,
    'ftell' : None,
    'gamma' : ('gamma', 'FUNC', 'math'),
    'gerror' : None,
    'getarg' : None,
    'get_command' : None,
    'get_command_argument' : None,
    'getcwd' : None,
    'getenv' : None,
    'get_environment_variable' : None,
    'getgid' : None,
    'getlog' : None,
    'getpid' : None,
    'getuid' : None,
    'gmtime' : None,
    'hostnm' : None,
    'huge' : None,
    'hypot' : ('hypot', 'FUNC', 'math'),
    'iachar' : None,
    'iall' : None,
    'iand' : None,
    'iany' : None,
    'iargc' : None,
    'ibclr' : None,
    'ibits' : None,
    'ibset' : None,
    'ichar' : None,
    'idate' : None,
    'ieor' : None,
    'ierrno' : None,
    'image_index' : None,
    'index' : None,
    'int' : ('int', 'FUNC', None),
    'int2' : None,
    'int8' : None,
    'ior' : None,
    'iparity' : None,
    'irand' : None,
    'is_iostat_end' : None,
    'is_iostat_eor' : None,
    'isatty' : None,
    'ishft' : None,
    'ishftc' : None,
    'isnan' : ('isnan', 'FUNC', 'math'),
    'itime' : None,
    'kill' : None,
    'kind' : None,
    'lbound' : None,
    'lcobound' : None,
    'leadz' : None,
    'len' : None,
    'len_trim' : None,
    'lge' : ('>=', 'INFIXOP', None),    # lexical string comparison
    'lgt' : ('>', 'INFIXOP', None),     # lexical string comparison
    'link' : None,
    'lle' : ('<=', 'INFIXOP', None),    # lexical string comparison
    'llt' : ('<', 'INFIXOP', None),     # lexical string comparison
    'lnblnk' : None,
    'loc' : None,
    'log' : ('log', 'FUNC', 'math'),
    'log10' : ('log10', 'FUNC', 'math'),
    'log_gamma' : ('lgamma', 'FUNC', 'math'),
    'logical' : None,
    'long' : None,
    'lshift' : None,
    'lstat' : None,
    'ltime' : None,
    'malloc' : None,
    'maskl' : None,
    'maskr' : None,
    'matmul' : None,
    'max' : ('max', 'FUNC', None),
    'maxexponent' : None,
    'maxloc' : None,
    'maxval' : None,
    'mclock' : None,
    'mclock8' : None,
    'merge' : None,
    'merge_bits' : None,
    'min' : ('min', 'FUNC', None),
    'minexponent' : None,
    'minloc' : None,
    'minval' : None,
    'mod' : ('%', 'INFIXOP', None),
    'modulo' : ('%', 'INFIXOP', None),
    'move_alloc' : None,
    'mvbits' : None,
    'nearest' : None,
    'new_line' : None,
    'nint' : None,
    'norm2' : None,
    'not' : None,
    'null' : None,
    'num_images' : None,
    'or' : None,
    'pack' : None,
    'parity' : None,
    'perror' : None,
    'popcnt' : None,
    'poppar' : None,
    'precision' : None,
    'present' : None,
    'product' : None,
    'radix' : None,
    'ran' : None,
    'rand' : None,
    'random_number' : None,
    'random_seed' : None,
    'range' : None,
    'rank ' : None,
    'real' : None,
    'rename' : None,
    'repeat' : None,
    'reshape' : None,
    'rrspacing' : None,
    'rshift' : None,
    'same_type_as' : None,
    'scale' : None,
    'scan' : None,
    'secnds' : None,
    'second' : None,
    'selected_char_kind' : None,
    'selected_int_kind' : None,
    'selected_real_kind' : None,
    'set_exponent' : None,
    'shape' : None,
    'shifta' : None,
    'shiftl' : None,
    'shiftr' : None,
    'sign' : None,
    'signal' : None,
    'sin' : ('sin', 'FUNC', 'math'),
    'sind' : None,
    'sinh' : ('sinh', 'FUNC', 'math'),
    'size' : None,
    'sizeof' : None,
    'sleep' : None,
    'spacing' : None,
    'spread' : None,
    'sqrt' : ('sqrt', 'FUNC', 'math'),
    'srand' : None,
    'stat' : None,
    'storage_size' : None,
    'sum' : None,
    'symlnk' : None,
    'system' : None,
    'system_clock' : None,
    'tan' : ('tan', 'FUNC', 'math'),
    'tand' : None,
    'tanh' : ('tanh', 'FUNC', 'math'),
    'this_image' : None,
    'time' : None,
    'time8' : None,
    'tiny' : None,
    'trailz' : None,
    'transfer' : None,
    'transpose' : None,
    'trim' : None,
    'ttynam' : None,
    'ubound' : None,
    'ucobound' : None,
    'umask' : None,
    'unlink' : None,
    'unpack' : None,
    'verify' : None,
    'xor' : ('^', 'INFIXOP', None)
}

