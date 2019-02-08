import sys
from fortran_format import *
from for2py_arrays import *

class mytype:
    def __init__(self):
        self.i : int = None
        self.a = Array([(1,3)])

def main():
    # var is a mytype object
    var = mytype()

    # x is an array of mytype objects: first allocate an Array object 
    # for x, then create a mytype object for each of its elements
    x = Array([(1,3)])
    for i in range(1,3+1):
        obj = mytype()
        x.set_(i, obj)

    var.i = 3

    for i in range(1,var.i+1):
        var.a.set_(i, var.i + i)
        for j in range(1,var.i+1):
            #mytype_obj = x.get_(i)
            #mytype_obj.a.set_(j, (i+j)/2.0)
            x.get_(i).a.set_(j, (i+j)/2.0)

    fmt_10 = Format(['I3', '4(3X, F5.3)'])
    for i in range(1,var.i+1):
        line = fmt_10.write_line([i, \
                                  var.a.get_(i), \
                                  x.get_(i).a.get_(1), \
                                  x.get_(i).a.get_(2), \
                                  x.get_(i).a.get_(3)])
        sys.stdout.write(line)


main()
