from typing import List

from python_mlir_toy.common import td, location


class ToyDialect(td.Dialect):
    def __init__(self, name: str):
        super().__init__(name)


class ToyOp(td.Op):
    def __init__(self, loc: location.Location, name: str):
        super().__init__(loc, name)


class ConstantOp(ToyOp):
    def __init__(self, loc: location.Location, operand):
        super().__init__(loc, 'toy.constant')
        self.operands: List[List[int]] = [operand]
        self.results: List[td.Tensor[int]] = [td.Tensor()]


class AddOp(ToyOp):
    def __init__(self, loc: location.Location, lhs: td.Tensor[int], rhs: td.Tensor[int]):
        super().__init__(loc, 'toy.add')
        self.operands: List[td.Tensor[int]] = [lhs, rhs]
        self.results = [td.Tensor()]


class FuncOp(ToyOp, td.IsolatedFromAbove):
    def __init__(self, loc: location.Location, name: str, function_type, body, arg_attrs=None, res_attrs=None):
        super().__init__(loc, 'toy.func')
        self.operands = [name, function_type, arg_attrs, res_attrs]
        self.regions = [body]


class GenericCallOp(ToyOp):
    def __init__(self, loc: location.Location, callee: str, inputs):
        super().__init__(loc, 'toy.generic_call')
        self.operands = [callee, inputs]
        self.results = [td.Tensor()]


class MulOp(ToyOp):
    def __init__(self, loc: location.Location, lhs: td.Tensor[int], rhs: td.Tensor[int]):
        super().__init__(loc, 'toy.mul')
        self.operands: List[td.Tensor[int]] = [lhs, rhs]
        self.results = [td.Tensor()]


class PrintOp(ToyOp):
    def __init__(self, loc: location.Location, operand: td.Tensor[int]):
        super().__init__(loc, 'toy.print')
        self.operands = [operand]


class ReshapeOp(ToyOp):
    def __init__(self, loc: location.Location, operand: td.Tensor[int]):
        super().__init__(loc, 'toy.reshape')
        self.operands = [operand]
        self.results = [td.Tensor()]


class ReturnOp(ToyOp, td.HasParent[FuncOp]):
    def __init__(self, loc: location.Location, operand: td.Tensor[int]):
        super().__init__(loc, 'toy.return')
        self.operands = [operand]


class TransposeOp(ToyOp):
    def __init__(self, loc: location.Location, operand: td.Tensor[int]):
        super().__init__(loc, 'toy.transpose')
        self.operands = [operand]
        self.results = [td.Tensor()]

