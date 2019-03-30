def get_S2_ranks(S2_mat):
    return [
        (val, r, c)
        for r, row in enumerate(S2_mat)
        for c, val in enumerate(row)
        if c > r
    ]


def get_min_s2_sensitivity(S2_mat):
    """
    Returns a tuple of the form:
        (S2-value, variable index 1, variable index 2)
    where S2-value is the minimum of the set of all S2 indices
    """
    return min(get_S2_ranks(S2_mat), key=lambda tup: abs(tup[0]))


def get_max_s2_sensitivity(S2_mat):
    """
    Returns a tuple of the form:
        (S2-value, variable index 1, variable index 2)
    where S2-value is the maximum of the set of all S2 indices
    """
    return max(get_S2_ranks(S2_mat), key=lambda tup: abs(tup[0]))
