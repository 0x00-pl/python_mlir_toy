import typing
from typing import List, Optional, Tuple

from python_mlir_toy.common import td, location, mlir_type, serializable, mlir_op, formater, scoped_text_parser, \
    scoped_text_printer
from python_mlir_toy.common.serializable import TextParser


class ToyOp(mlir_op.Op):
    def __init__(self, loc: location.Location, operands=None, result_types=None, blocks=None):
        super().__init__(loc, operands, result_types, blocks)


class ConstantOp(ToyOp):
    op_name = 'toy.constant'

    def __init__(self, loc: location.Location, shape: List[int], values: List[float]):
        result_type = mlir_type.F64TensorType() if len(shape) == 0 else mlir_type.RankedF64TensorType(shape)
        super().__init__(loc, result_types=[result_type])
        self.shape = shape
        self.values = values

    @classmethod
    def get_assembly_format(cls) -> formater.Format:
        def _print_op(obj, dst: scoped_text_printer.ScopedTextPrinter):
            operand_names = (dst.lookup_value_name(item) for item in obj.operands)
            cls._operands_format.print(operand_names, dst)
            result_type_list = list(item.ty for item in obj.results)
            cls._results_ty_format.print(result_type_list, dst)

        def _parse_op(src: scoped_text_parser.ScopedTextParser):
            literal = cls._literal_format.parse(src)
            loc = cls._location_format.parse(src)
            return cls(loc, literal.shape, literal.values)

        return formater.CustomFormat(_print_op, _parse_op)


class ToyFuncOp(ToyOp, mlir_op.FuncOp):
    op_name = 'toy.func'

    def __init__(self, loc: location.Location, function_name: str,
                 arg_name_list: List[Tuple[str, location.Location] | str], function_type: mlir_type.FunctionType,
                 block: mlir_op.Block):
        super().__init__(loc, blocks=[block])
        self.function_name = function_name
        self.function_type = function_type
        self.arg_name_list = arg_name_list
        assert len(arg_name_list) == len(function_type.inputs)


class GenericCallOp(ToyOp):
    op_name = 'toy.generic_call'

    def __init__(self, loc: location.Location, callee: ToyFuncOp, *inputs: td.Value):
        super().__init__(loc, operands=list(inputs), result_types=callee.function_type.outputs)
        self.callee = callee
        # todo: verify callee input types
        assert len(inputs) == len(callee.function_type.inputs)

    @classmethod
    def get_assembly_format(cls) -> formater.Format:
        def _print_op(obj: GenericCallOp, dst: scoped_text_printer.ScopedTextPrinter):
            assert isinstance(obj, GenericCallOp)
            cls._function_name_format.print(obj.callee.function_name, dst)
            dst.print('(')
            operands_name = [dst.lookup_value_name(item) for item in obj.operands]
            cls._operands_format.print(operands_name, dst)
            dst.print(')', ' : ')
            obj.callee.function_type.print(dst)
            cls._location_format.print(obj.location, dst)

        def _parse_op(src: scoped_text_parser.ScopedTextParser):
            callee_name = cls._function_name_format.parse(src)
            callee_value = src.lookup_var(callee_name)
            assert isinstance(callee_value, td.ConstantValue)
            assert isinstance(callee_value.ty, mlir_type.FunctionType)
            src.drop_token('(')
            operands_name = cls._operands_format.parse(src)
            operands = [src.lookup_var(operands_name) for operands_name in operands_name]
            src.drop_token(')')
            src.drop_token(':')
            function_type = mlir_type.parse_function_type(src)
            loc = cls._location_format.parse(src)
            return cls(loc, callee_value.value, *operands)

        return formater.CustomFormat(_print_op, _parse_op)

    def function_name_format(self):
        def printer(dst: serializable.TextPrinter):
            dst.print(self.callee.function_name)

        def parser(src: TextParser):
            raise NotImplementedError

        return printer, parser

    def function_type_format(self):
        def printer(dst: serializable.TextPrinter):
            self.callee.function_type.print(dst)

        def parser(src: TextParser):
            raise NotImplementedError

        return printer, parser


class AddOp(ToyOp):
    op_name = 'toy.add'

    def __init__(self, loc: location.Location, lhs: td.Value, rhs: td.Value):
        super().__init__(loc, operands=[lhs, rhs], result_types=[mlir_type.F64TensorType()])
        assert mlir_type.F64TensorType() <= lhs.ty
        assert mlir_type.F64TensorType() <= rhs.ty


class MulOp(ToyOp):
    op_name = 'toy.mul'

    def __init__(self, loc: location.Location, lhs: td.Value, rhs: td.Value):
        super().__init__(loc, operands=[lhs, rhs], result_types=[mlir_type.F64TensorType()])
        assert mlir_type.F64TensorType() <= lhs.ty
        assert mlir_type.F64TensorType() <= rhs.ty

    @classmethod
    def build_as_generic_op(cls, loc: location.Location, operands: typing.List[td.Value] = None,
                            result_types: typing.List[mlir_type.Type] = None, blocks=None):
        assert len(operands) == 2
        assert len(result_types) == 1 or result_types is None
        assert blocks is None
        return cls(loc, operands[0], operands[1])


class PrintOp(ToyOp):
    op_name = 'toy.print'

    def __init__(self, loc: location.Location, operand: td.Value):
        super().__init__(loc, operands=[operand])
        assert mlir_type.F64TensorType() <= operand.ty

    @classmethod
    def get_assembly_format(cls) -> formater.Format:
        def _print_op(obj, dst: scoped_text_printer.ScopedTextPrinter):
            assert isinstance(obj, PrintOp)
            operand_name = dst.lookup_value_name(obj.operands[0])
            cls._variable_name_format.print(operand_name, dst)
            cls._location_format.print(obj.location, dst)

        def _parse_op(src: scoped_text_parser.ScopedTextParser):
            operand_name = cls._variable_name_format.parse(src)
            operand = src.lookup_var(operand_name)
            loc = cls._location_format.parse(src)
            return cls(loc, operand)

        return formater.CustomFormat(_print_op, _parse_op)


class ReshapeOp(ToyOp):
    op_name = 'toy.reshape'

    def __init__(self, loc: location.Location, shape: List[int], operand: td.Value):
        super().__init__(loc, operands=[operand], result_types=[mlir_type.RankedF64TensorType(shape)])

    @classmethod
    def get_assembly_format(cls) -> formater.Format:
        def _print_op(obj, dst: scoped_text_printer.ScopedTextPrinter):
            operand_name = dst.lookup_value_name(obj.operands[0])
            assert operand_name is not None
            dst.print('(', operand_name, ' : ')
            obj.operands[0].ty.print(dst)
            dst.print(')', ' to ')
            obj.results[0].ty.print(dst)
            cls._location_format.print(obj.location, dst)

        def _parse_op(src: scoped_text_parser.ScopedTextParser):
            src.drop_token('(')
            operand_name = cls._variable_name_format.parse(src)
            operand = src.lookup_var(operand_name)
            src.drop_token(':')
            operand.ty.parse(src)
            src.drop_token(')')
            src.drop_token('to')
            result_type = mlir_type.parse_type(src)
            assert isinstance(result_type, mlir_type.RankedTensorType)
            shape = result_type.shape
            loc = cls._location_format.parse(src)
            return cls(loc, shape, operand)

        return formater.CustomFormat(_print_op, _parse_op)


class ReturnOp(ToyOp, td.HasParent[ToyFuncOp]):
    op_name = 'toy.return'

    def __init__(self, loc: location.Location, operand: Optional[td.Value] = None):
        super().__init__(loc, operands=([operand]) if operand is not None else [])

    @classmethod
    def get_assembly_format(cls) -> formater.Format:
        def _print_op(obj, dst: scoped_text_printer.ScopedTextPrinter):
            if len(obj.operands) > 0:
                operand_names = (dst.lookup_value_name(item) for item in obj.operands)
                cls._operands_format.print(operand_names, dst)
                result_type_list = list(item.ty for item in obj.results)
                cls._results_ty_format.print(result_type_list, dst)

        def _parse_op(src: scoped_text_parser.ScopedTextParser):
            if src.last_token() == '%':
                operand_name = cls._variable_name_format.parse(src)
                operand = src.lookup_var(operand_name)
                _ = cls._results_ty_format.parse(src)
            else:
                operand = None

            loc = cls._location_format.parse(src)
            return cls(loc, operand)

        return formater.CustomFormat(_print_op, _parse_op)


class TransposeOp(ToyOp):
    op_name = 'toy.transpose'

    # class Format(formater.Format):
    #     def __init__(self):
    #         self.format_list = [formater.ConstantStrFormat('('), mlir_op.Op._variable_name_format,
    #                             formater.ConstantStrFormat(':'), formater.TypeFormat(), formater.ConstantStrFormat(')'),
    #                             formater.ConstantStrFormat(' to '), formater.TypeFormat(),
    #                             formater.OptionalFormat(formater.LocationFormat(), (lambda loc: loc is not None),
    #                                                     'loc')]
    #
    #     def print(self, obj, dst: serializable.TextPrinter):
    #         assert isinstance(obj, TransposeOp)
    #         for fmt in self.format_list:
    #             fmt.print(obj, dst)
    #
    #     def parse(self, src: scoped_text_parser.ScopedTextParser):
    #         _, operand_name, _, operand_type, _, _, result_type, loc = [item.parse(src) for item in self.format_list]
    #         operand = src.lookup_var(operand_name)
    #         assert operand is not None
    #         return TransposeOp(loc, [1, 0], operand)
    #
    # _format = Format()

    def __init__(self, loc: location.Location, permutation: List[int], operand: td.Value):
        if isinstance(operand.ty, mlir_type.RankedTensorType):
            shape = operand.ty.shape
            result_type = mlir_type.RankedF64TensorType([shape[i] for i in permutation])
        else:
            result_type = mlir_type.F64TensorType()
        super().__init__(loc, operands=[operand], result_types=[result_type])

    @classmethod
    def get_assembly_format(cls) -> formater.Format:
        def _print_op(obj, dst: scoped_text_printer.ScopedTextPrinter):
            assert isinstance(obj, TransposeOp)
            dst.print('(')
            cls._variable_name_format.print(dst.lookup_value_name(obj.operands[0]), dst)
            dst.print(':')
            cls._type_format.print(obj.operands[0].ty, dst)
            dst.print(')', ' to ')
            cls._type_format.print(obj.results[0].ty, dst)
            cls._location_format.print(obj.location, dst)

        def _parse_op(src: scoped_text_parser.ScopedTextParser):
            src.drop_token('(')
            operand_name = cls._variable_name_format.parse(src)
            operand = src.lookup_var(operand_name)
            assert operand is not None
            src.drop_token(':')
            operand_type = cls._type_format.parse(src)
            assert operand.ty <= operand_type
            src.drop_token(')')
            src.drop_token('to')
            result_type = cls._type_format.parse(src)
            assert isinstance(result_type, mlir_type.TensorType)
            loc = cls._location_format.parse(src)
            return TransposeOp(loc, [1, 0], operand)

        return formater.CustomFormat(_print_op, _parse_op)