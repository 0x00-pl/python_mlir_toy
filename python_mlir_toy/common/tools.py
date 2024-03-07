
def with_sep(lst, sep):
    first = True
    for i in lst:
        if first:
            first = False
        else:
            sep()
        yield i
