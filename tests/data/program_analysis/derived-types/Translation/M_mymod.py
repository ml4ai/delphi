# Python module for Fortran module mymod

def foo_int(x, result):
    result[0] = 47 * x[0] + 23
    return

def foo_real(x, result):
    result[0] = int(x[0]) * 31 + 17
    return

def foo_bool(x, result):
    if x[0]:
        result[0] = 937
    else:
        result[0] = -732
    return

def foo(x, result):
    if isinstance(x[0], bool) and isinstance(result[0], int):
        foo_bool(x, result)
    elif isinstance(x[0], int) and isinstance(result[0], int):
        foo_int(x, result)
    elif isinstance(x[0], float) and isinstance(result[0], int):
        foo_real(x, result)
    else:
        sys.stderr.write('ERROR: no types matched foo()')
        sys.exit(1)


