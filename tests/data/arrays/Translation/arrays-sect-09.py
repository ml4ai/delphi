from delphi.translators.for2py.scripts.fortran_format import *
from delphi.translators.for2py.scripts.for2py_arrays import *

def main():
    A = Array([(1,5)])
    B = Array([(1,10)])

    for i in range(1,5+1):
        A.set_(i, 0)          # A(i) = 0
        B.set_(i, 11*i)       # B(i) = 11*i
        B.set_(i+5, 7*i+6)    # B(i+5) = 7*i+6

    fmt_obj = Format(['5(I5)'])

    sys.stdout.write(fmt_obj.write_line([A.get_(1), A.get_(2), A.get_(3), A.get_(4), A.get_(5)]))

    B_elems = flatten(implied_loop_expr((lambda x:x), 2, 8, 3))
    B_vals = B.get_elems(B_elems)

    A_elems = flatten(implied_loop_expr((lambda x:x), 1, 5, 2))
    A.set_elems(A_elems, B_vals)

    sys.stdout.write(fmt_obj.write_line([A.get_(1), A.get_(2), A.get_(3), A.get_(4), A.get_(5)]))


main()
