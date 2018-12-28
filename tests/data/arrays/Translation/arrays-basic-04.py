from fortran_format import *
from for2py_arrays import *

def main():
    arr = Array([(1,5),(1,5)])

    for i in range(1,5+1):
        for j in range(1,5+1):
            arr.set((i,j), i+j)

    fmt_obj = Format(['5(I5,X)'])

    for i in range(1,5+1):
        sys.stdout.write(fmt_obj.write_line([arr.get((i,1)),
                                             arr.get((i,2)),
                                             arr.get((i,3)),
                                             arr.get((i,4)),
                                             arr.get((i,5))]))


main()
