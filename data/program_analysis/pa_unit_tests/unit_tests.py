
# ------------------------------------------------
# [unit 1] assignment literal

x = 3


# ------------------------------------------------
# [unit 2] assignment with source variable (lambda)

x = y + 2


# ------------------------------------------------
# [unit 3] conditional 1: single var condition

if x:
    y = 1
else:
    y = 2


# ------------------------------------------------
# [unit 4] conditional 2: inline condition, no var

if x < 3:
    y = 1
else:
    y = 2


# ------------------------------------------------
# [unit 5] conditional 3: inline condition, with var

if x < y + 2:
    y = 1
else:
    x = 2


# ------------------------------------------------
# [unit 6] loop 1:

for x in range(0, 2):
    y = x


# ------------------------------------------------
# [unit 7] container 1: simple, no arg, no return

def fn7():
    x = 1


# ------------------------------------------------
# [unit 8] container 2: simple, arg, no return
# NOTE: from DBN perspective, y is same as x

def fn8(x):
    y = x


# ------------------------------------------------
# [unit 9] container 3: simple, no arg, return

def fn9():
    y = 3
    return y


# ------------------------------------------------
# [unit 10] container calling container, no return

def fn10_1(x):
    y = x


def fn10_2():
    fn10_1(3)


# ------------------------------------------------
# [unit 11] container calling container, with return and assign
# NOTE: this is ultimately just an assignment of a literal to z -- should that be identified?

def fn11_1(x):
    y = x
    return y


def fn11_2():
    z = fn11_1(3)


# ------------------------------------------------
# [unit 12] container with loop calling container, with return and assing

def fn12_1(x):
    y = x
    return y


def fn12_2():
    for d in range(4):
        y = fn12_1(d)


# ------------------------------------------------
# [unit 13] container with conditional return

def fn13(x):
    if x < 3:
        return 2
    else:
        return 4


# ------------------------------------------------
# [unit 14] container calling container with conditional return and assignment

def fn14_1(x):
    if x < 3:
        return 2
    else:
        return 4


def fn14_2():
    y = fn14_1(3)


# ------------------------------------------------
# [unit 15] container with multiple value return

def fn15():
    return 2, 3


# ------------------------------------------------
# [unit 16] container calling container with multiple value return and assignment

def fn16_1():
    return 2, 3


def fn16_2():
    x, y = fn16_1()

