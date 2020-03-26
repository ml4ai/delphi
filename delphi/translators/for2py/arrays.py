"""
    File: for2py_arrays.py
    Purpose: Code to handle array manipulation in the Python code generated
    by for2py.

    Usage: see the document "for2py: Handling Fortran Arrays"
"""

import copy
import itertools
import sys
from . import For2PyError

_GET_ = 0
_SET_ = 1

################################################################################
#                                                                              #
#                                 Array objects                                #
#                                                                              #
################################################################################


class Array:
    """ bounds is a list [(lo1,hi1), (lo2,hi2), ..., (loN, hiN)] of pairs of
        lower and upper bounds for the dimensions of the array.  The length
        of the list bounds gives the number of dimensions of the array.
    """
    def __init__(self, types, bounds):
        self._bounds = bounds
        self._types = types
        self._values = self._mk_uninit_array(bounds)

    def _mk_uninit_array(self, bounds):
        """ given a list of bounds for the N dimensions of an array, 
            _mk_uninit_array() creates and returns an N-dimensional array of 
            the size specified by the bounds with each element set to the value
            None."""
        if len(bounds) == 0:
            raise For2PyError("Zero-length arrays current not handled!.")

        this_dim = bounds[0]
        lo, hi = this_dim[0], this_dim[1]
        sz = hi-lo+1

        if len(bounds) == 1:
            return [None] * sz

        sub_array = self._mk_uninit_array(bounds[1:])
        this_array = [copy.deepcopy(sub_array) for _ in range(sz)]

        return this_array

    def bounds(self):
        """bounds() returns a list of pairs (lo,hi) giving the lower and upper
        bounds of the array."""
        return self._bounds

    def lower_bd(self, i):
        """lower_bd(i) returns the lower bound of the array in dimension i.
           Dimensions are numbered from 0 up."""
        this_dim_bounds = self._bounds[i]
        return this_dim_bounds[0]

    def upper_bd(self, i):
        """upper_bd(i) returns the upper bound of the array in dimension i.
           Dimensions are numbered from 0 up."""
        this_dim_bounds = self._bounds[i]
        return this_dim_bounds[1]

    def _posn(self, bounds, idx):
        """given bounds = (lo,hi) and an index value idx, _posn(bounds, idx)
           returns the position in a 0-based array corresponding to idx in 
           the (lo,hi)-based array.  It generates an error if idx < lo or 
           idx > hi."""
        lo, hi = bounds[0], bounds[1]
        assert lo <= idx <= hi, f"Array index {idx} out of bounds: {bounds}\n"
   
        return idx-lo

    def _access(self, subs, acc_type, val):
        """_access(subs, acc_type, val) accesses the array element specified by
           the tuple of subscript values, subs.  If acc_type == _GET_ it 
           returns the value of this element; else it sets this element to the
           value of the argument val."""
        if isinstance(subs, int):  
            # if subs is just an integer, take it to be an index value.
            subs = (subs,)

        if len(subs) == 0:
            raise For2PyError("Zero-length arrays currently not handled.")

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

    def set_(self, subs, val):
        """set_() sets the value of the array element specified by the given
           tuple of array subscript values to the argument val."""
        self._access(subs, _SET_, val)

    def get_(self, subs):
        """get_() returns the value of the array element specified by the 
           given tuple of array subscript values."""
        return self._access(subs, _GET_, None)

    def get_elems(self, subs_list):
        """get_elems(subs_list) returns a list of values of the array elements
           specified by the list of subscript values subs_list (each element of
           subs_list is a tuple of subscripts identifying an array element)."""
        return [self.get_(subs) for subs in subs_list]

    def set_elems(self, subs, vals):
        """set_elems(subs, vals) sets the array elements specified by the list
           of subscript values subs (each element of subs is a tuple of 
           subscripts identifying an array element) to the corresponding value
           in vals."""
        if isinstance(vals, (int, float)):
            # if vals is a scalar, extend it to a list of appropriate length
            vals = [vals] * len(subs)

        for i in range(len(subs)):
            self.set_(subs[i], vals[i])

    def get_sum(self):
        """Calculates the sum of all values in the array.
        """
        arr_bounds = self._bounds
        summed_val = 0

        low = arr_bounds[0][0]+1
        up = arr_bounds[0][1]+1
        for idx in range(low, up):
            arr_element = self.get_(idx)
            # Multi-dimensional array.
            if type(arr_element) == list:
                for elem in arr_element:
                    if elem is not None:
                        summed_val += elem
            # Single-dimensional array.
            else:
                assert (
                        type(arr_element) == int
                        or type(arr_element) == float
                ), f"Only numbers can be summed. Array element type: {type(arr_element)}"
                summed_val += arr_element

        return summed_val

    def get_type(self):
        return self._types


################################################################################
#                                                                              #
#                    Functions for accessing parts of arrays                   #
#                                                                              #
################################################################################

def all_subs(bounds):
    """given a list of tuples specifying the bounds of an array, all_subs()
       returns a list of all the tuples of subscripts for that array."""
    idx_list = []
    for i in range(len(bounds)):
        this_dim = bounds[i]
        lo, hi = this_dim[0], this_dim[1]   # bounds for this dimension
        this_dim_idxs = range(lo, hi+1)    # indexes for this dimension
        idx_list.append(this_dim_idxs)

    return idx2subs(idx_list)


def idx2subs(idx_list):
    """Given a list idx_list of index values for each dimension of an array, 
       idx2subs() returns a list of the tuples of subscripts for all of the 
       array elements specified by those index values.

       Note: This code adapted from that posted by jfs at
       https://stackoverflow.com/questions/533905/get-the-cartesian-product-
       of-a-series-of-lists"""
    if not idx_list:
        return [()]
    return [items + (item,) 
            for items in idx2subs(idx_list[:-1]) for item in idx_list[-1]]


def array_values(expr):
    """Given an expression expr denoting a list of values, array_values(expr) 
       returns a list of values for that expression."""
    if isinstance(expr, Array):
        return expr.get_elems(all_subs(expr._bounds))
    elif isinstance(expr, list):
        vals = [array_values(x) for x in expr]
        return flatten(vals)
    else:
        return [expr]


def array_subscripts(expr):
    """Given a subscript expression expr (i.e., an expression that denotes the
       set of elements of some array that are to be accessed), 
       array_subscripts() returns a list of the elements denoted by expr."""
    if isinstance(expr, Array):
        return all_subs(expr._bounds)
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

def flatten(in_list):
    """given a list of values in_list, flatten returns the list obtained by 
       flattening the top-level elements of in_list."""
    out_list = []
    for val in in_list:
        if isinstance(val, list):
            out_list.extend(val)
        else:
            out_list.append(val)

    return out_list


def implied_loop_expr(expr, start, end, delta):
    """given the parameters of an implied loop -- namely, the start and end 
       values together with the delta per iteration -- implied_loop_expr() 
       returns a list of values of the lambda expression expr applied to 
       successive values of the implied loop."""
    if delta > 0:
        stop = end+1
    else:
        stop = end-1

    result_list = [expr(x) for x in range(start, stop, delta)]

    # return the flattened list of results
    return list(itertools.chain(result_list))
