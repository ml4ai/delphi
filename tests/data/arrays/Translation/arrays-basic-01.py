from fortran_format import *
from for2py_arrays import *

def main():
    arr = Array([(1,10)])

    for i in range(1,10+1):
        arr.set((i,), i*i)

    fmt_obj = Format(['I5'])

    for i in range(1,10+1):
        val = arr.get((i,))
        sys.stdout.write(fmt_obj.write_line([val]))

main()
