def nim(n, k):
    pieces = n
    if k < 1:
        raise Exception("k cannot be lower than 1")
    maximum_removal = k
    while pieces > 0:
