"""
    File: for2py_arrays.py
    Purpose: Code to handle array manipulation in the Python code generated
        by for2py.

    Usage:
"""

import sys
import copy
import itertools

_GET_ = 0
_SET_ = 1

################################################################################
#                                                                              #
#                                 Array objects                                #
#                                                                              #
################################################################################

class Array:
    # bounds is a list [(lo1,hi1), (lo2,hi2), ..., (loN, hiN)] of pairs of
    # lower and upper bounds for the dimensions of the array.  The length
    # of the list bounds gives the number of dimensions of the array.
    def __init__(self, bounds):
        self._bounds = bounds
        self._values = self._mk_uninit_array(bounds)


    # given a list of bounds for the N dimensions of an array, _mk_uninit_array()
    # creates and returns an N-dimensional array of the size specified by the
    # bounds with each element set to the value None.
    def _mk_uninit_array(self, bounds):
        if len(bounds) == 0:
            sys.stderr.write("Zero-length arrays current not handled!\n")
            sys.exit(1)

        this_dim = bounds[0]
        lo,hi = this_dim[0],this_dim[1]
        sz = hi-lo+1

        if len(bounds) == 1:
            return [None] * sz

        sub_array = self._mk_uninit_array(bounds[1:])
        this_array = [copy.deepcopy(sub_array) for i in range(sz)]

        return this_array


    # bounds() returns a list of pairs (lo,hi) giving the lower and upper
    # bounds of the array.
    def bounds(self):
        return self._bounds


    # lower_bd(i) returns the lower bound of the array in dimension i.  Dimensions
    # are numbered from 0 up.
    def lower_bd(self, i):
        this_dim_bounds = self._bounds[i]
        return this_dim_bounds[0]


    # upper_bd(i) returns the upper bound of the array in dimension i.  Dimensions
    # are numbered from 0 up.
    def upper_bd(self, i):
        this_dim_bounds = self._bounds[i]
        return this_dim_bounds[1]


    # given bounds = (lo,hi) and an index value idx, _posn(bounds, idx)
    # returns the position in a 0-based array corresponding to idx in the 
    # (lo,hi)-based array.  It generates an error if idx < lo or idx > hi.
    def _posn(self, bounds, idx):
        lo,hi = bounds[0],bounds[1]
        if idx < lo or idx > hi:
            sys.stderr.write("Array index (value = {:d}) out of bounds {}\n".\
                                 format(idx, str(bounds)))
            sys.exit(1)

        return idx-lo
   

    # _access(subs, acc_type, val) accesses the array element specified by the 
    # tuple of subscript values, subs.  If acc_type == __GET it returns the value 
    # of this element; else it sets this element to the value of the argument val.
    def _access(self, subs, acc_type, val):
        # if subs is just an integer, take it to be an index value.
        if isinstance(subs, int):
            subs = (subs,)

        if len(subs) == 0:
            sys.stderr.write("get/set: Zero-length arrays currently not handled!\n")
            sys.exit(1)

        bounds = self._bounds
        sub_arr = self._values
        ndims = len(subs)
        for i in range(ndims):
            this_pos = self._posn(bounds[i], subs[i])

            if i == ndims-1:
                if acc_type == _GET_:
                    return sub_arr[this_pos]
                else:
                    sub_arr[this_pos] = val
            else:
                sub_arr = sub_arr[this_pos]


    # set() sets the value of the array element specified by the given tuple
    # of array subscript values to the argument val.
    def set(self, subs, val):
        self._access(subs, _SET_, val)


    # get() returns the value of the array element specified by the given tuple
    # of array subscript values.
    def get(self, subs):
        return self._access(subs, _GET_, None)


    # get_elems(subs_list) returns a list of values of the array elements specified
    # by the list of subscript values subs_list (each element of subs_list is a
    # tuple of subscripts identifying an array element).  
    def get_elems(self, subs_list):
        return [self.get(subs) for subs in subs_list]


    # set_elems(subs, vals) sets the array elements specified by the list
    # of subscript values subs (each element of subs is a tuple of subscripts
    # identifying an array element) to the corresponding value in vals.
    def set_elems(self, subs, vals):
        # if vals is a scalar, extend it to a list of appropriate length
        if isinstance(vals, (int, float)):
            vals = [vals] * len(subs)

        for i in range(len(subs)):
            self.set(subs[i], vals[i])


    # given a list of tuples specifying the bounds of an array, all_subs()
    # returns a list of all the tuples of subscripts for that array.
    def all_subs(self, bounds):

        def all_subs_1(self, bounds):
            this_dim = bounds[0]
            lo,hi = this_dim[0],this_dim[1]   # bounds for this dimension
            this_dim_subs = range(lo,hi+1)    # subscripts for this dimension
        
            if len(bounds) == 1:
                return [[x] for x in this_dim_subs]
            else:
                rest_dim_subs = self.all_subs_1(bounds[1:])
                subs_list = []
                for x in this_dim_subs:
                    for y in rest_dim_subs:
                        subs_list.append(y.insert(0,x))
    
                return subs_list

        return list(map(tuple, all_subs_1(self,bounds)))


################################################################################
#                                                                              #
#                    Functions for accessing parts of arrays                   #
#                                                                              #
################################################################################

# Given an expression expr denoting a list of values, values(expr) returns 
# a list of values for that expression.
def values(expr):
    if isinstance(expr, Array):
        return expr.get_elems(expr.all_subs(expr._bounds))
    elif isinstance(expr, list):
        vals = [values(x) for x in expr]
        return flatten(vals)
    else:
        return [expr]


# Given a subscript expression expr (i.e., an expression that denotes the
# set of elements of some array that are to be accessed), subscripts() returns
# a list of the elements denoted by expr.
def subscripts(expr):
    if isinstance(expr, Array):
        return expr.all_subs(expr._bounds)
    elif isinstance(expr, list):
        subs = [subscripts(x) for x in expr]
        return flatten(subs)
    else:
        return [expr]


################################################################################
#                                                                              #
#                               Assorted utilities                             #
#                                                                              #
################################################################################

# given a list of values in_list, flatten returns the list obtained by flattening
# the top-level elements of in_list.
def flatten(in_list):
    out_list = []
    for val in in_list:
        if isinstance(val, list):
            out_list.extend(val)
        else:
            out_list.append(val)

    return out_list


# given the parameters of an implied loop -- namely, the start and end values
# together with the delta per iteration -- implied_loop_expr() returns a list 
# of values of the lambda expression expr applied to successive values of the
# implied loop.
def implied_loop_expr(expr, start, end, delta):
    # Simulating 1-based indexing in a language with 0-based indexing is a p.i.t.a.
    if delta > 0:
        stop = end+1
    else:
        stop = end-1

    result_list = [expr(x) for x in range(start,stop,delta)]

    # return the flattened list of results
    return list(itertools.chain(result_list))


