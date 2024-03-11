from typing import List, Optional, Tuple

from python_mlir_toy.common import td, location, mlir_type, serializable, mlir_op, scoped_text_printer, tools


class ToyOp(mlir_op.Op):
    def __init__(self, loc: location.Location, name: str, operands=None, result_types=None, blocks=None):
        super().__init__(loc, name, operands, result_types, blocks)


class ConstantOp(ToyOp):
    def __init__(self, loc: location.Location, shape: List[int], values: List[float]):
        result_type = mlir_type.F64TensorType() if len(shape) == 0 else mlir_type.RankedF64TensorType(shape)
        super().__init__(loc, 'toy.constant', result_types=[result_type])
        self.shape = shape
        self.values = values

    def print(self, dst: scoped_text_printer.ScopedTextPrinter):
        self.print_return_values(dst)
        dst.print(self.name)
        dst.print(f'dense<{self.values}> :')
        self.results[0].ty.print(dst)
        dst.print()
        self.print_loc(dst)
        dst.print_newline()

    def print_content(self, dst: serializable.TextPrinter):
        dst.print(f'dense<{self.values}>')
        self.location.print(dst)


class FuncOp(ToyOp, td.IsolatedFromAbove):
    def __init__(self, loc: location.Location, function_name: str,
                 arg_name_list: List[Tuple[str, location.Location] | str], function_type: mlir_type.FunctionType,
                 block: mlir_op.Block):
        super().__init__(loc, 'toy.func', blocks=[block])
        self.function_name = function_name
        self.function_type = function_type
        self.arg_name_list = arg_name_list
        assert len(arg_name_list) == len(function_type.inputs)

    def get_operand_types(self):
        return self.function_type.inputs

    def get_result_types(self):
        return self.function_type.outputs

    def print_content(self, dst: scoped_text_printer.ScopedTextPrinter):
        dst.print(self.function_name, end='')
        argument_dict = {}
        with dst:
            dst.print('(', end='')
            for idx, arg_value in enumerate(tools.with_sep(self.blocks[0].arguments, lambda: dst.print(','))):
                if isinstance(self.arg_name_list[idx], str):
                    arg_name = dst.next_unused_symbol('%arg')
                    arg_loc = None
                elif isinstance(self.arg_name_list[idx], tuple):
                    arg_name = dst.next_unused_symbol('%arg')
                    arg_loc = self.arg_name_list[idx][1]

                dst.insert_value_name(arg_value, arg_name)
                argument_dict[arg_name] = arg_value
                dst.print(arg_name, ': ', sep='', end='')
                arg_value.ty.print(dst)
                if arg_loc is not None:
                    dst.print()
                    arg_loc.print(dst)
            dst.print(')')
        if len(self.function_type.outputs) > 0:
            dst.print('->')
            if len(self.function_type.outputs) == 1:
                self.function_type.outputs[0].print(dst)
                dst.print()
            else:
                dst.print('(', end='')
                for result_type in tools.with_sep(self.function_type.outputs, lambda: dst.print(',')):
                    result_type.print(dst)
                    dst.print()
                dst.print(')')

        dst.print('{', end='\n')
        with dst:
            for name, value in argument_dict.items():
                dst.insert_value_name(value, name)
            self.blocks[0].print(dst)
        dst.print_ident()
        dst.print('}', end='')


class GenericCallOp(ToyOp):
    def __init__(self, loc: location.Location, callee: FuncOp, *inputs: td.Value):
        super().__init__(loc, 'toy.generic_call', operands=list(inputs), result_types=callee.get_result_types())
        self.callee = callee
        # todo: verify callee input types
        assert len(inputs) == len(callee.get_operand_types())

    def print(self, dst: scoped_text_printer.ScopedTextPrinter):
        self.print_return_values(dst)
        dst.print(self.name)
        dst.print('@', self.callee.function_name, sep='', end='')
        self.print_arguments_detail(dst)
        dst.print(' : ', end='')
        self.callee.function_type.print(dst)
        self.print_loc(dst)
        dst.print_newline()


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

    def print(self, dst: scoped_text_printer.ScopedTextPrinter):
        self.print_return_values(dst)
        dst.print(self.name, end='')
        dst.print(f'({dst.lookup_value_name(self.operands[0])} :')
        self.operands[0].ty.print(dst)
        dst.print(') to ', end='')
        self.results[0].ty.print(dst)
        dst.print()
        self.print_loc(dst)
        dst.print_newline()


class ReturnOp(ToyOp, td.HasParent[FuncOp]):
    def __init__(self, loc: location.Location, operand: Optional[td.Value] = None):
        super().__init__(loc, 'toy.return', operands=([operand]) if operand is not None else [])

    def print(self, dst: scoped_text_printer.ScopedTextPrinter):
        self.print_return_values(dst)
        dst.print(self.name)
        if len(self.operands) > 0:
            dst.print(dst.lookup_value_name(self.operands[0]))
            dst.print(':')
            self.operands[0].ty.print(dst)
            dst.print()
        self.print_loc(dst)
        dst.print_newline()


class TransposeOp(ToyOp):
    def __init__(self, loc: location.Location, permutation: List[int], operand: td.Value):
        if isinstance(operand.ty, mlir_type.RankedTensorType):
            shape = operand.ty.shape
            result_type = mlir_type.RankedF64TensorType([shape[i] for i in permutation])
        else:
            result_type = mlir_type.F64TensorType()
        super().__init__(loc, 'toy.transpose', operands=[operand], result_types=[result_type])

    def print(self, dst: scoped_text_printer.ScopedTextPrinter):
        self.print_return_values(dst)
        dst.print(self.name, end='')
        dst.print(f'({dst.lookup_value_name(self.operands[0])} :')
        self.operands[0].ty.print(dst)
        dst.print(') to ', end='')
        self.results[0].ty.print(dst)
        dst.print()
        self.print_loc(dst)
        dst.print_newline()
