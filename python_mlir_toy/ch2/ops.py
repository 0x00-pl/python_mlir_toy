from typing import List, Optional

from python_mlir_toy.common import td, location, mlir_type


# class ToyDialect(td.Dialect):
#     def __init__(self, name: str):
#         super().__init__(name)


class ToyOp(td.Op):
    def __init__(self, loc: location.Location, name: str, operands=None, result_types=None, blocks=None):
        super().__init__(loc, name, operands, result_types, blocks)


class ConstantOp(ToyOp):
    def __init__(self, loc: location.Location, shape: List[int], values: List[float]):
        super().__init__(loc, 'toy.constant', result_types=[mlir_type.F64TensorType()])
        self.shape = shape
        self.values = values


class FuncOp(ToyOp, td.IsolatedFromAbove):
    def __init__(self, loc: location.Location, name: str, function_type, block: td.Block):
        super().__init__(loc, 'toy.func', blocks=[block])
        self.name = name
        self.function_type = function_type

    def get_operand_types(self):
        return self.function_type[0]

    def get_result_types(self):
        return self.function_type[1]


class GenericCallOp(ToyOp):
    def __init__(self, loc: location.Location, callee, *inputs: td.Value):
        super().__init__(loc, 'toy.generic_call', operands=list(inputs), result_types=callee.get_result_types())
        # todo: verify callee input types
        assert len(inputs) == len(callee.get_operand_types())


class AddOp(ToyOp):
    def __init__(self, loc: location.Location, lhs: td.Value, rhs: td.Value):
        super().__init__(loc, 'toy.add', operands=[lhs, rhs], result_types=[mlir_type.F64TensorType()])
        assert mlir_type.F64TensorType() <= lhs.ty
        assert mlir_type.F64TensorType() <= rhs.ty


class MulOp(ToyOp):
    def __init__(self, loc: location.Location, lhs: td.Value, rhs: td.Value):
        super().__init__(loc, 'toy.mul', operands=[lhs, rhs], result_types=[mlir_type.F64TensorType()])
        assert mlir_type.F64TensorType() <= lhs.ty
        assert mlir_type.F64TensorType() <= rhs.ty


class PrintOp(ToyOp):
    def __init__(self, loc: location.Location, operand: td.Value):
        super().__init__(loc, 'toy.print', operands=[operand])
        assert mlir_type.F64TensorType() <= operand.ty


class ReshapeOp(ToyOp):
    def __init__(self, loc: location.Location, shape: List[int], operand: td.Value):
        super().__init__(loc, 'toy.reshape', operands=[operand],
                         result_types=[mlir_type.RankedF64TensorType(shape)])


class ReturnOp(ToyOp, td.HasParent[FuncOp]):
    def __init__(self, loc: location.Location, operand: Optional[td.Value] = None):
        super().__init__(loc, 'toy.return', operands=([operand]) if operand is not None else [])


class TransposeOp(ToyOp):
    def __init__(self, loc: location.Location, permutation: List[int], operand: td.Value):
        if isinstance(operand.ty, mlir_type.RankedTensorType):
            shape = operand.ty.shape
            result_type = mlir_type.RankedF64TensorType([shape[i] for i in permutation])
        else:
            result_type = mlir_type.F64TensorType()
        super().__init__(loc, 'toy.transpose', operands=[operand], result_types=[result_type])
