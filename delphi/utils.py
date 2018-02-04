from itertools import repeat, accumulate, islice, chain
from functools import reduce
from tqdm import tqdm
from typing import TypeVar, Iterator, Tuple, Callable, Iterable, List, Any

T = TypeVar('T')

def prepend(x: T, xs: Iterable[T]) -> Iterator[T]:
    """ Prepend a value to an iterable. """
    return chain([x], xs)

def append(x: T, xs: Iterable[T]) -> Iterator[T]:
    """ Append a value to an iterable. """
    return chain(xs, [x])

def scanl(f: Callable[[T, T], T], x: T, xs: Iterable[T]) -> Iterator[T]:
    return accumulate(prepend(x, xs), f)

def scanl1(f: Callable[[T, T], T], xs: Iterable[T]) -> Iterator[T]:
    return accumulate(xs, f)

def foldl(f: Callable[[T, T], T], x: T, xs: Iterable[T]) -> T:
    return reduce(f, xs, x)

def foldl1(f: Callable[[T, T], T], xs: Iterable[T]) -> T:
    return reduce(f, xs)

def flatten(l: Iterable) -> Iterable:
    return sum(map(flatten,l),[]) if isinstance(l,list) or isinstance(l, tuple) else [l]

def iterate(f: Callable[[T], T], x0: T) -> Iterator[T]:
    return scanl(lambda x, _: f(x), x0, repeat(None))

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
