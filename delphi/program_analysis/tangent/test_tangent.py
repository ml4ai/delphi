import tangent


def f(x):
    a = x * x
    b = x * a
    c = a + b
    return c


print('f(3.0)', f(3.0))


df = tangent.grad(f, verbose=1)

print('df(3.0)', df(3.0))


# -------------------------------------------------------------------
print('-----------------------------------')


def g(x):
    c = (x * x) + (x * x * x)
    return c


print('g(3.0)', g(3.0))

dg = tangent.grad(g, verbose=1)

print('dg(3.0)', dg(3.0))

