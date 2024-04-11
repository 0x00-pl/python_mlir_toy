import typing

from python_mlir_toy.common import mlir_type

T = typing.TypeVar('T')


class Value:
    def __init__(self, ty: mlir_type.Type):
        self.ty = ty


class ConstantValue(Value, typing.Generic[T]):
    def __init__(self, ty: mlir_type.Type, value: T):
        super().__init__(ty)
        self.value = value


class IsolatedFromAbove:
    def __init__(self):
        pass


class HasParent(typing.Generic[T]):
    def __init__(self):
        self.ty = T
