def foo(x, y):
    a = x + 3
    for i in range(10):
        x += i
        if x > 5:
            break
        y += (a * x)
    return x, y


foo(5, 3)
