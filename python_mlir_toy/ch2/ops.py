from typing import List

from python_mlir_toy.common import td


class ToyDialect(td.Dialect):
    def __init__(self, name: str):
        super().__init__(name)


class ToyOp(td.Op):
    def __init__(self, name: str):
        super().__init__(name)


class ConstantOp(ToyOp):
    def __init__(self, operand):
        super().__init__('toy.constant')
        self.operands: List[List[int]] = [operand]
        self.results: List[td.Tensor[int]] = [td.Tensor()]


class AddOp(ToyOp):
    def __init__(self, lhs: td.Tensor[int], rhs: td.Tensor[int]):
        super().__init__('toy.add')
        self.operands: List[td.Tensor[int]] = [lhs, rhs]
        self.results = [td.Tensor()]


class FuncOp(ToyOp, td.IsolatedFromAbove):
    def __init__(self, name: str, function_type, body, arg_attrs=None, res_attrs=None):
        super().__init__('toy.func')
        self.operands = [name, function_type, arg_attrs, res_attrs]
        self.regions = [body]


class GenericCallOp(ToyOp):
    def __init__(self, callee: str, inputs):
        super().__init__('toy.generic_call')
        self.operands = [callee, inputs]
        self.results = [td.Tensor()]


class MulOp(ToyOp):
    def __init__(self, lhs: td.Tensor[int], rhs: td.Tensor[int]):
        super().__init__('toy.mul')
        self.operands: List[td.Tensor[int]] = [lhs, rhs]
        self.results = [td.Tensor()]


class PrintOp(ToyOp):
    def __init__(self, operand: td.Tensor[int]):
        super().__init__('toy.print')
        self.operands = [operand]


class ReshapeOp(ToyOp):
    def __init__(self, operand: td.Tensor[int]):
        super().__init__('toy.reshape')
        self.operands = [operand]
        self.results = [td.Tensor()]


class ReturnOp(ToyOp, td.HasParent[FuncOp]):
    def __init__(self, operand: td.Tensor[int]):
        super().__init__('toy.return')
        self.operands = [operand]


class TransposeOp(ToyOp):
    def __init__(self, operand: td.Tensor[int]):
        super().__init__('toy.transpose')
        self.operands = [operand]
        self.results = [td.Tensor()]

