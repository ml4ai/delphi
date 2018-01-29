from itertools import repeat, accumulate, islice, chain
from functools import reduce
from tqdm import tqdm

prepend = lambda x, xs: chain([x], xs)
append  = lambda x, xs: chain(xs, [x])
scanl   = lambda f, x, xs: accumulate(prepend(x, xs), f)
scanl1  = lambda f, xs: accumulate(xs, f)
foldl   = lambda f, x, xs: reduce(f, xs, x)
foldl1  = lambda f, xs: reduce(f, xs)
iterate = lambda f, x0: scanl(lambda x, _: f(x), x0, repeat(None))
flatten = lambda l: sum(map(flatten,l),[]) if isinstance(l,list) or isinstance(l, tuple) else [l]
take    = lambda n, xs: islice(xs, n)
ptake   = lambda n, xs: tqdm(take(n, xs), total = n)
ltake   = lambda n, xs: list(take(n, xs))
lmap    = lambda f, xs: list(map(f, xs))
lfilter = lambda f, xs: list(filter(f, xs))
lzip    = lambda *xs: list(zip(*xs))
compose = lambda *fs: foldl1(lambda f, g: lambda x: f(g(x)), fs)
flatMap = lambda f, xs: flatten(lmap(f, xs))
