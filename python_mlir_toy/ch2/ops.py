import typing
from typing import Optional

from python_mlir_toy.common import td, location, mlir_type, mlir_op, mlir_literal, bounded_format


class ToyOp:
    pass


class ConstantOp(mlir_op.Op, ToyOp):
    op_name = 'toy.constant'

    def __init__(
            self, loc: location.Location, literal: mlir_literal.Literal,
            output_types: typing.List[mlir_type.Type] = None
    ):
        super().__init__(loc)
        output_type = literal.get_type()
        if output_types is not None:
            assert len(output_types) == 1
            assert output_type <= output_types[0]
        self.literal = literal
        self.output = td.Value(output_type)

    def get_inputs(self) -> typing.List[td.Value]:
        return []

    def get_outputs(self) -> typing.List[td.Value]:
        return [self.output]

    @classmethod
    def get_format_list(cls):
        return [bounded_format.BoundedLiteralAttrFormat('literal'), bounded_format.OutputsTypeFormat(),
                bounded_format.LocationFormat()]


class ToyFuncOp(mlir_op.FuncOp, ToyOp):
    op_name = 'toy.func'


class ToyGenericCallOp(mlir_op.GenericCallOp, ToyOp):
    op_name = 'toy.generic_call'


class AddOp(mlir_op.BinaryOp, ToyOp):
    op_name = 'toy.add'


class MulOp(mlir_op.BinaryOp, ToyOp):
    op_name = 'toy.mul'


class PrintOp(mlir_op.Op, ToyOp):
    op_name = 'toy.print'

    def __init__(self, loc: location.Location, operand: td.Value, operand_type: mlir_type.Type = None):
        super().__init__(loc)
        self.operand = operand
        assert operand_type is None or operand.ty <= operand_type

    def get_inputs(self) -> typing.List[td.Value]:
        return [self.operand]

    def get_outputs(self) -> typing.List[td.Value]:
        return []

    @classmethod
    def get_format_list(cls):
        return [bounded_format.BoundedInputFormat('operand'), bounded_format.BoundedTypeFormat('operand'),
                bounded_format.LocationFormat()]


class ReshapeOp(mlir_op.Op, ToyOp):
    op_name = 'toy.reshape'

    # _op_name_format = formater.NamespacedSymbolFormat(end='')

    def __init__(
            self, loc: location.Location, operand: td.Value, operand_type: mlir_type.Type = None,
            output_type: mlir_type.Type = None
    ):
        super().__init__(loc)
        self.operand = operand
        assert isinstance(self.operand.ty, mlir_type.RankedTensorType)
        assert operand_type is None or self.operand.ty <= operand_type
        self.output = td.Value(output_type)

    def get_inputs(self) -> typing.List[td.Value]:
        return [self.operand]

    def get_outputs(self) -> typing.List[td.Value]:
        return [self.output]

    @classmethod
    def get_format_list(cls):
        return [bounded_format.ConstantStrFormat('('), bounded_format.BoundedInputFormat('operand'),
                bounded_format.BoundedTypeFormat('operand'), bounded_format.ConstantStrFormat(')'),
                bounded_format.ConstantStrFormat('to'), bounded_format.BoundedTypeFormat('output', prefix=None),
                bounded_format.LocationFormat()]


class ReturnOp(mlir_op.Op, ToyOp):
    op_name = 'toy.return'

    def __init__(
            self, loc: location.Location, operand: Optional[td.Value] = None,
            operand_type: Optional[mlir_type.Type] = None
    ):
        super().__init__(loc)
        self.operand = operand
        assert operand_type is None or operand.ty <= operand_type

    def get_inputs(self) -> typing.List[td.Value]:
        return [self.operand] if self.operand is not None else []

    def get_outputs(self) -> typing.List[td.Value]:
        return []

    @classmethod
    def get_format_list(cls):
        return [bounded_format.BoundedOptionalInputFormat('operand'), bounded_format.BoundedTypeFormat('operand'),
                bounded_format.LocationFormat()]


class TransposeOp(mlir_op.Op, ToyOp):
    op_name = 'toy.transpose'

    def __init__(
            self, loc: location.Location, operand: td.Value, operand_type: mlir_type.Type = None,
            output_type: mlir_type.Type = None
    ):
        super().__init__(loc)
        assert operand_type is None or operand.ty <= operand_type
        if isinstance(operand.ty, mlir_type.RankedTensorType):
            assert len(operand.ty.shape) >= 2
            *shape, m2, m1 = operand.ty.shape
            new_shape = [*shape, m1, m2]
            result_type = mlir_type.RankedF64TensorType(new_shape)
        else:
            result_type = mlir_type.F64TensorType()
        self.operand = operand
        assert output_type is None or result_type <= output_type
        self.output = td.Value(result_type)

    def get_inputs(self) -> typing.List[td.Value]:
        return [self.operand]

    def get_outputs(self) -> typing.List[td.Value]:
        return [self.output]

    @classmethod
    def get_format_list(cls):
        return [bounded_format.ConstantStrFormat('('), bounded_format.BoundedInputFormat('operand'),
                bounded_format.BoundedTypeFormat('operand'), bounded_format.ConstantStrFormat(')'),
                bounded_format.ConstantStrFormat('to'), bounded_format.BoundedTypeFormat('output', prefix=None),
                bounded_format.LocationFormat()]
