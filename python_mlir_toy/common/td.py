import typing

from python_mlir_toy.common import location, mlir_type

T = typing.TypeVar('T')


# class Dialect:
#     def __init__(self, name: str):
#         self.name = name


class Value:
    def __init__(self, ty: mlir_type.Type):
        self.ty = ty


class Block:
    def __init__(self, input_types=None):
        self.arguments = [Value(ty) for ty in input_types] if input_types else []
        self.op_list: typing.List['Op'] = []

    def add_ops(self, op_list: typing.List['Op']):
        self.op_list.extend(op_list)


class Op:
    def __init__(self, loc: location.Location, name: str, operands: typing.List[Value] = None,
                 result_types: typing.List[mlir_type.Type] = None, blocks: typing.List[Block] = None):
        self.location = loc
        self.name = name
        self.operands = operands if operands else []
        self.results = [Value(ty) for ty in result_types] if result_types else []
        self.blocks = blocks


class IsolatedFromAbove:
    def __init__(self):
        pass


class HasParent(typing.Generic[T]):
    def __init__(self):
        self.ty = T
