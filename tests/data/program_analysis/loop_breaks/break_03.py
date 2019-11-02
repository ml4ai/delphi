def foo(x, y):
    a = x + 3
    for i in range(10):
        x += i
        if x > 5:
            break
        if i < 5:
            a = 2
            y += (a * i)
        else:
            break
    return x, y


foo(5, 3)
