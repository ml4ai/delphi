from delphi.translators.for2py.format import *
from delphi.translators.for2py.static_save import static_vars


@static_vars(["w"])
def f(n, x):
    if n[0] == 0:
        f.w = 111
    else:
        f.w = 2 * f.w
    x[0] = f.w


@static_vars(["w"])
def g(n, x):
    if n[0] == 0:
        g.w = 999
    else:
        g.w = g.w // 3  # ***
    x[0] = g.w


def main():
    a, b = [None], [None]

    f([0], a)
    g([0], b)
    fmt_10 = Format(['"a = "', 'I5', '";   b = "', 'I5'])
    line = fmt_10.write_line([a[0], b[0]])
    sys.stdout.write(line)

    f([1], a)
    g([1], b)
    line = fmt_10.write_line([a[0], b[0]])
    sys.stdout.write(line)

    f([1], a)
    g([1], b)
    line = fmt_10.write_line([a[0], b[0]])
    sys.stdout.write(line)


main()
