from typing import List, Optional

from python_mlir_toy.common import td, location, mlir_type, serializable, mlir_op, scoped_text_printer, tools


class ToyOp(mlir_op.Op):
    def __init__(self, loc: location.Location, name: str, operands=None, result_types=None, blocks=None):
        super().__init__(loc, name, operands, result_types, blocks)


class ConstantOp(ToyOp):
    def __init__(self, loc: location.Location, shape: List[int], values: List[float]):
        super().__init__(loc, 'toy.constant', result_types=[mlir_type.F64TensorType()])
        self.shape = shape
        self.values = values

    def print_content(self, dst: serializable.TextPrinter):
        dst.print(f'dense<{self.values}>')
        self.location.print(dst)


class FuncOp(ToyOp, td.IsolatedFromAbove):
    def __init__(self, loc: location.Location, function_name: str, function_type: mlir_type.FunctionType,
                 block: mlir_op.Block):
        super().__init__(loc, 'toy.func', blocks=[block])
        self.function_name = function_name
        self.function_type = function_type

    def get_operand_types(self):
        return self.function_type.inputs

    def get_result_types(self):
        return self.function_type.outputs

    def print_content(self, dst: scoped_text_printer.ScopedTextPrinter):
        dst.print(self.function_name)
        argument_dict = {}
        with dst.indent:
            dst.print('(', end='')
            for arg_value in tools.with_sep(self.blocks[0].arguments, lambda: dst.print(',')):
                arg_name = dst.next_unused_symbol('%arg')
                dst.insert_value_name(arg_value, arg_name)
                argument_dict[arg_name] = arg_value
                dst.print(arg_name, ':', sep='', end='')
                arg_value.ty.print(dst)
            dst.print(')')
        dst.print('->')
        if len(self.function_type.outputs) == 1:
            self.function_type.outputs[0].print(dst)
            dst.print()
        else:
            dst.print('(', end='')
            for result_value in tools.with_sep(self.function_type.outputs, lambda: dst.print(',')):
                result_value.ty.print(dst)
                dst.print()
            dst.print(')')

        with dst.indent:
            for name, value in argument_dict.items():
                dst.insert_value_name(value, name)
            self.blocks[0].print(dst)


class GenericCallOp(ToyOp):
    def __init__(self, loc: location.Location, callee, *inputs: td.Value):
        super().__init__(loc, 'toy.generic_call', operands=list(inputs), result_types=callee.get_result_types())
        # todo: verify callee input types
        assert len(inputs) == len(callee.get_operand_types())

    def print_content(self, dst: serializable.TextPrinter):
        pass


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

    def print_content(self, dst: serializable.TextPrinter):
        pass


class PrintOp(ToyOp):
    def __init__(self, loc: location.Location, operand: td.Value):
        super().__init__(loc, 'toy.print', operands=[operand])
        assert mlir_type.F64TensorType() <= operand.ty

    def print_content(self, dst: serializable.TextPrinter):
        pass


class ReshapeOp(ToyOp):
    def __init__(self, loc: location.Location, shape: List[int], operand: td.Value):
        super().__init__(loc, 'toy.reshape', operands=[operand], result_types=[mlir_type.RankedF64TensorType(shape)])

    def print_content(self, dst: serializable.TextPrinter):
        pass


class ReturnOp(ToyOp, td.HasParent[FuncOp]):
    def __init__(self, loc: location.Location, operand: Optional[td.Value] = None):
        super().__init__(loc, 'toy.return', operands=([operand]) if operand is not None else [])

    def print_content(self, dst: serializable.TextPrinter):
        pass


class TransposeOp(ToyOp):
    def __init__(self, loc: location.Location, permutation: List[int], operand: td.Value):
        if isinstance(operand.ty, mlir_type.RankedTensorType):
            shape = operand.ty.shape
            result_type = mlir_type.RankedF64TensorType([shape[i] for i in permutation])
        else:
            result_type = mlir_type.F64TensorType()
        super().__init__(loc, 'toy.transpose', operands=[operand], result_types=[result_type])

    def print_content(self, dst: serializable.TextPrinter):
        pass
