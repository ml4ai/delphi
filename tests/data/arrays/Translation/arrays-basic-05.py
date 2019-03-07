from delphi.translators.for2py.format import *
from delphi.translators.for2py.arrays import *

def main():
    A = Array([(-3,1),(-4,0)])

    for i in range(-3,1+1):
        for j in range(-4,0+1):
            A.set_((i,j), i+j)

    fmt_obj = Format(['5(I5,X)'])

    for i in range(-3,1+1):
        sys.stdout.write(fmt_obj.write_line([A.get_((i,-4)),
                                             A.get_((i,-3)),
                                             A.get_((i,-2)),
                                             A.get_((i,-1)),
                                             A.get_((i,0))]))


main()
