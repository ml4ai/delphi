from delphi.translators.for2py.fortran_format import *
from delphi.translators.for2py.for2py_arrays import *


def main():
    arr = Array([(1,10)])

    # values from the implied loop (11*I, I = 1,3)
    sublist1 = implied_loop_expr((lambda x: 11*x), 1, 3, 1)

    # values from the implied loop (10*I-1, I = 4,7)
    sublist2 = implied_loop_expr((lambda x: 10*x-1), 4, 7, 1)

    # values from the implied loop (I*I, I = 8,10)
    sublist3 = implied_loop_expr((lambda x: x*x), 8, 10, 1)

    arr_constr = flatten([sublist1, sublist2, sublist3])
    arr_subs = array_subscripts(arr)

    arr.set_elems(arr_subs, arr_constr)

    fmt_obj = Format(['I5'])

    for i in range(1,10+1):
        val = arr.get_((i,))
        sys.stdout.write(fmt_obj.write_line([val]))

main()
