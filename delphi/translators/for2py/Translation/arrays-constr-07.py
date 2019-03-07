from fortran_format import *
from for2py_arrays import *


def main():
    A = Array([(1,12)])
    B = Array([(1,12)])
    C = Array([(1,12)])

    B_constr = [12,11,10,9,8,7,6,5,4,3,2,1]
    B_subs = array_subscripts(B)
    B.set_elems(B_subs, B_constr)      

    C_constr = [1,3,5,7,9,11,13,15,17,19,21,23]
    C_subs = array_subscripts(C)
    C.set_elems(C_subs, C_constr)      

    A_constr = implied_loop_expr((lambda x: C.get_(B.get_(x))),12,1,-1)
    A_subs = array_subscripts(A)
    A.set_elems(A_subs, A_constr)

    fmt_obj = Format(['I5'])

    for i in range(1,12+1):
        val = A.get_((i,))
        sys.stdout.write(fmt_obj.write_line([val]))

main()
