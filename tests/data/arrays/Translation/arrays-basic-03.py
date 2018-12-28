from fortran_format import *
from for2py_arrays import *

def main():
    arr = Array([(1,10)])
    idx = Array([(1,10)])

    for i in range(1,10+1):
        arr.set((i,), i*i)

    for i in range(1,5+1):
        idx.set((i,), 2*i)
        idx.set((i+5,), 2*i-1)

    fmt_obj = Format(['I5'])

    for i in range(1,10+1):
        val = arr.get([idx.get((i,))])
        sys.stdout.write(fmt_obj.write_line([val]))

main()
