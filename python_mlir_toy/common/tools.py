import typing

T = typing.TypeVar('T')


def with_sep(lst: typing.Collection[T], sep: typing.Callable[[], typing.Any]) -> typing.Generator[T, None, None]:
    first = True
    for i in lst:
        if first:
            first = False
        else:
            sep()
        yield i
