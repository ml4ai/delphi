from delphi.translators.for2py.scripts.fortran_format import *
from delphi.translators.for2py.scripts.for2py_arrays import *

def main():
    arr = Array([(1,10)])

    for i in range(1,10+1):
        arr.set_((i,), i*i)

    fmt_obj = Format(['I5'])

    for i in range(1,10+1):
        val = arr.get_((i,))
        sys.stdout.write(fmt_obj.write_line([val]))

main()
