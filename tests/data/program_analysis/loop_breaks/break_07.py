def foo(x, y):
    a = x + 3
    for i in range(10):
        x += random_number()
        if not (x > 5):
            y += (a * x)
        else:
            break

        if i == 8:
            return x-1, y-1
        else:
            if i == 9:
                return x-2, y-2

    x = max(x, 10)
    y = min(y, 5)
    return x, y


foo(5, 3)
