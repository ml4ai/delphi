from fortran_format import *
from for2py_arrays import *

def main():
    A = Array([(1,5)])
    B = Array([(1,3)])

    # B = (/2,3,4/)
    B.set_elems(subscripts(B), [2,3,4])

    for i in range(1,5+1):
        A.set(i, 0)

    fmt_obj = Format(['5(I5)'])

    sys.stdout.write(fmt_obj.write_line([A.get(1), A.get(2), A.get(3), A.get(4), A.get(5)]))

    A.set_elems(values(B), 17)

    sys.stdout.write(fmt_obj.write_line([A.get(1), A.get(2), A.get(3), A.get(4), A.get(5)]))


main()
