""" Helper functions for functional programming.  """

from itertools import repeat, accumulate, islice, chain
from functools import reduce
from tqdm import tqdm
from typing import TypeVar, Iterator, Tuple, Callable, Iterable, List, Any, Union

T = TypeVar('T')
U = TypeVar('U')


def prepend(x: T, xs: Iterable[T]) -> Iterator[T]:
    """ Prepend a value to an iterable. 

    Parameters
    ----------
    x
        An element of type T.
    xs
        An iterable of elements of type T. 

    Returns
    -------
    Iterator
        An iterator that yields *x* followed by elements of *xs*.

    Examples
    --------
    
    >>> from delphi.utils import prepend
    >>> list(prepend(1, [2, 3]))
    [1, 2, 3]

    """
    return chain([x], xs)


def append(x: T, xs: Iterable[T]) -> Iterator[T]:
    """ Append a value to an iterable.

    Parameters
    ----------
    x
        An element of type T.
    xs
        An iterable of elements of type T. 

    Returns
    -------
    Iterator
        An iterator that yields elements of *xs*, then yields *x*.


    Examples
    --------
    >>> from delphi.utils import append
    >>> list(append(1, [2, 3]))
    [2, 3, 1]

    """

    return chain(xs, [x])


def scanl(f: Callable[[T, U], T], x: T, xs: Iterable[U]) -> Iterator[T]:
    """ Make an iterator that returns accumulated results of a binary function
    applied to elements of an iterable.

    .. math::
        scanl(f, x_0, [x_1, x_2, ...]) = [x_0, f(x_0, x_1), f(f(x_0, x_1), x_2), ...]

    Parameters
    ----------
    f
        A binary function of two arguments of type T.
    x
        An initializer element of type T.
    xs
        An iterable of elements of type T. 

    Returns
    -------
    Iterator
        The iterator of accumulated results.


    Examples
    --------
    >>> from delphi.utils import scanl
    >>> list(scanl(lambda x, y: x + y, 10, range(5)))
    [10, 10, 11, 13, 16, 20]

    """

    return accumulate(prepend(x, xs), f)


def scanl1(f: Callable[[T, T], T], xs: Iterable[T]) -> Iterator[T]:
    """ Make an iterator that returns accumulated results of a binary function
    applied to elements of an iterable.

    .. math::
        scanl1(f, [x_0, x_1, x_2, ...]) = [x_0, f(x_0, x_1), f(f(x_0, x_1), x_2), ...]

    Parameters
    ----------
    f
        A binary function of two arguments of type T.
    xs
        An iterable of elements of type T. 

    Returns
    -------
    Iterator
        The iterator of accumulated results.


    Examples
    --------
    >>> from delphi.utils import scanl1
    >>> list(scanl1(lambda x, y: x + y, range(5)))
    [0, 1, 3, 6, 10]

    """
    return accumulate(xs, f)


def foldl(f: Callable[[T, U], T], x: T, xs: Iterable[U]) -> T:
    """ Returns the accumulated result of a binary function applied to elements
    of an iterable.

    .. math::
        foldl(f, x_0, [x_1, x_2, x_3]) = f(f(f(f(x_0), x_1), x_2), x_3)


    Examples
    --------
    >>> from delphi.utils import foldl
    >>> foldl(lambda x, y: x + y, 10, range(5))
    20

    """
    return reduce(f, xs, x)


def foldl1(f: Callable[[T, T], T], xs: Iterable[T]) -> T:
    """ Returns the accumulated result of a binary function applied to elements
    of an iterable.

    .. math::
        foldl1(f, [x_0, x_1, x_2, x_3]) = f(f(f(f(x_0), x_1), x_2), x_3)


    Examples
    --------
    >>> from delphi.utils import foldl1
    >>> foldl1(lambda x, y: x + y, range(5))
    10
    """

    return reduce(f, xs)


def flatten(xs: Union[List, Tuple]) -> List:
    """ Flatten a nested list or tuple. """
    return sum(map(flatten, xs),[]) if (isinstance(xs,list) 
                                     or isinstance(xs, tuple)) else [xs]


def iterate(f: Callable[[T], T], x: T) -> Iterator[T]:
    """ Makes infinite iterator that returns the result of successive
    applications of a function to an element

    .. math::
        iterate(f, x) = [x, f(x), f(f(x)), f(f(f(x))), ...]

    Examples
    --------
    >>> from delphi.utils import iterate, take
    >>> list(take(5, iterate(lambda x: x*2, 1)))
    [1, 2, 4, 8, 16]
    """
    return scanl(lambda x, _: f(x), x, repeat(None))


def take(n: int, xs: Iterable[T]) -> Iterable[T]:
    return islice(xs, n)


def ptake(n: int, xs: Iterable[T]) -> Iterable[T]:
    """ take with a tqdm progress bar. """
    return tqdm(take(n, xs), total = n)


def ltake(n: int, xs: Iterable[T]) -> List[T]:
    """ A non-lazy version of take. """
    return list(take(n, xs))


def lmap(f: Callable[[T], T], xs: Iterable[T]) -> List[T]:
    """ A non-lazy version of map. """
    return list(map(f, xs))


def lfilter(f: Callable[[T], T], xs: Iterable[T]) -> List[T]:
    """ A non-lazy version of filter. """
    return list(filter(f, xs))


def lzip(*xs: Iterable[Iterable])->List[Iterable]:
    """ A non-lazy version of zip. """
    return list(zip(*xs))


def compose(*fs: Any) -> Callable:
    """ Compose functions from left to right.

    e.g. compose(f, g)(x) = f(g(x))
    """
    return foldl1(lambda f, g: lambda x: f(g(x)), fs)


def flatMap(f: Callable, xs: Iterable) -> List:
    """ Map a function onto an iterable and flatten the result. """
    return flatten(lmap(f, xs))
