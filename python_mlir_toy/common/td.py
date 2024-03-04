import typing

from python_mlir_toy.common import location

T = typing.TypeVar('T')


class Dialect:
    def __init__(self, name: str):
        self.name = name


class Op:
    def __init__(self, loc: location.Location, name: str):
        self.location = loc
        self.name = name
        self.operands = None
        self.results = None
        self.regions = None


class Tensor(typing.Generic[T]):
    def __init__(self):
        self.ty = T


class Region:
    def __init__(self, blocks=None):
        self.blocks = blocks


class IsolatedFromAbove:
    def __init__(self):
        pass


class HasParent(typing.Generic[T]):
    def __init__(self):
        self.ty = T

