
# ------------------------------------------------
# [unit 1] assignment literal

x = 3


# ------------------------------------------------
# [unit 2] assignment with source variable (lambda)

x = y + 2


# ------------------------------------------------
# [unit 3] conditional: single var condition

if x:
    y = 1
else:
    y = 2


# ------------------------------------------------
# [unit 4] conditional: inline condition, no var

if x < 3:
    y = 1
else:
    y = 2


# ------------------------------------------------
# [unit 5] conditional: inline condition, with var

if x < y + 2:
    y = 1
else:
    x = 2


# ------------------------------------------------
# [unit 6] loop: contains assign fn that references loop-index

for x in range(0, 2):
    y = x


# ------------------------------------------------
# [unit 7] container: simple, no arg, no return

def fn():
    x = 1


# ------------------------------------------------
# [unit 8] container: simple, arg, no return
# NOTE: from DBN perspective, y is same as x

def fn(x):
    y = x


# ------------------------------------------------
# [unit 9] container: simple, no arg, return var

def fn():
    y = 3
    return y


# ------------------------------------------------
# [unit 10] container: simple, no arg, return with assign

def fn():
    y = 3
    return y + 3


# ------------------------------------------------
# [unit 11] container calling container, no return

def fn_a(x):
    y = x


def fn_b():
    fn_a(3)


# ------------------------------------------------
# [unit 12] container calling container, with return and assign
# NOTE: this is ultimately just an assignment of a literal to z -- should that be identified?

def fn_a(x):
    y = x
    return y


def fn_b():
    z = fn_a(3)


# ------------------------------------------------
# [unit 13] container with loop calling container, with return and assing

def fn_a(x):
    y = x
    return y


def fn_b():
    for d in range(4):
        y = fn_a(d)


# ------------------------------------------------
# [unit 14] container with conditional return

def fn(x):
    if x < 3:
        return 2
    else:
        return 4


# ------------------------------------------------
# [unit 15] container calling container with conditional return and assignment

def fn_a(x):
    if x < 3:
        return 2
    else:
        return 4


def fn_b():
    y = fn_a(3)


# ------------------------------------------------
# [unit 16] container with multiple value return

def fn():
    return 2, 3


# ------------------------------------------------
# [unit 17] container calling container with multiple value return and assignment

def fn_a():
    return 2, 3


def fn_b():
    x, y = fn_a()

