import sys
import typing

from python_mlir_toy.common import location, td, serializable, scoped_text_printer, mlir_type, tools, scoped_text_parser, formater
from python_mlir_toy.common.serializable import TextPrinter, TextParser


class TypeFormat(formater.Format):
    def print(self, obj, dst: serializable.TextPrinter):
        assert isinstance(obj, mlir_type.Type)
        obj.print(dst)

    def parse(self, src: serializable.TextParser):
        return mlir_type.parse_type(src)


class TypeListFormat(formater.Format):
    def __init__(self, parentheses_required: bool):
        self.types_format = formater.RepeatFormat(TypeFormat(), ', ')
        self.parentheses_required = parentheses_required

    def print(self, obj, dst: serializable.TextPrinter):
        assert isinstance(obj, list)
        if len(obj) != 1 or self.parentheses_required:
            dst.print('(')
            self.types_format.print(obj, dst)
            dst.print(')')
        else:
            self.types_format.print(obj, dst)

    def parse(self, src: serializable.TextParser) -> typing.List[typing.Any]:
        if src.last_char() == '(':
            src.process_token()
            ret = self.types_format.parse(src)
            src.process_token(')')
        else:
            ret = self.types_format.parse(src)
        return ret


class OpFormat(formater.Format):
    def __init__(self, op_cls):
        self.operands_format = formater.RepeatFormat(formater.StrFormat(), ', ')
        self.results_format = formater.OptionalFormat(
            formater.ListFormat([formater.ConstantStrFormat(' : '), TypeListFormat(parentheses_required=False)]),
            (lambda result_type_list: len(result_type_list) > 0),
            ':'
        )
        self.op_cls: typing.Type[Op] = op_cls

    def print(self, obj, dst: scoped_text_printer.ScopedTextPrinter):
        assert isinstance(obj, Op)
        operand_names = (dst.lookup_value_name(item) for item in obj.operands)
        self.operands_format.print(operand_names, dst)
        result_type_list = list(item.ty for item in obj.results)
        self.results_format.print(result_type_list, dst)

    def parse(self, src: scoped_text_parser.ScopedTextParser):
        loc = location.FileLineColLocation(*src.last_location())
        operand_names = self.operands_format.parse(src)
        operands = [src.lookup_var(operand_name) for operand_name in operand_names]
        result_types = self.results_format.parse(src)
        return self.op_cls.build(loc, operands, result_types, None)




#
#
# class OperandNameFormat(serializable.TextSerializable):
#     def __init__(self, op: 'Op', idx: int):
#         self.op = op
#         self.idx = idx
#
#     def print(self, dst: scoped_text_printer.ScopedTextPrinter):
#         operand_value = self.op.operands[self.idx]
#         operand_name = dst.lookup_value_name(operand_value)
#         dst.print(operand_name, end='')
#
#     @classmethod
#     def parse(cls, src: TextParser):
#         raise NotImplementedError('TODO')
#
#
# class OperandTypeFormat(serializable.TextSerializable):
#     def __init__(self, op: 'Op', idx: int):
#         self.op = op
#         self.idx = idx
#
#     def print(self, dst: scoped_text_printer.ScopedTextPrinter):
#         operand_value = self.op.operands[self.idx]
#         operand_value.ty.print(dst)
#
#     @classmethod
#     def parse(cls, src: TextParser):
#         raise NotImplementedError('TODO')
#
#
# class OpOperandsFormat:
#     @staticmethod
#     def print(op: 'Op', dst: scoped_text_printer.ScopedTextPrinter):
#         for operand in tools.with_sep(op.operands, lambda: dst.print(', ', end='')):
#             operand_name = dst.lookup_value_name(operand)
#             dst.print(operand_name, end='')
#
#     @staticmethod
#     def parse(src: scoped_text_parser.ScopedTextParser):
#         assert src.last_token_kind() == serializable.TokenKind.Identifier
#         operand_name = src.last_token()
#         src.process_token()
#         operand_value = src.lookup_var(operand_name)
#         operands = [operand_value]
#         while src.last_token() == ',':
#             src.process_token()
#             operand_name = src.last_token()
#             operand_value = src.lookup_var(operand_name)
#             operands.append(operand_value)
#
#         def set_operands(op: Op):
#             op.operands = operands
#
#         return set_operands
#
#
# class OpResultTypeFormat:
#     @staticmethod
#     def print(op: 'Op', dst: scoped_text_printer.ScopedTextPrinter):
#         if len(op.results) > 1:
#             dst.print('(', end='')
#             for result_value in tools.with_sep(op.results, lambda: dst.print(',')):
#                 result_value.ty.print(dst)
#             dst.print(')', end='')
#         elif len(op.results) == 1:
#             op.results[0].ty.print(dst)
#         else:
#             dst.print('none', end='')
#
#     @staticmethod
#     def parse(src: scoped_text_parser.ScopedTextParser):
#         result_types = []
#         if src.last_token() == '(':
#             src.process_token()
#             ty_name = src.last_token()  # todo: parse type
#             src.process_token()
#             result_types.append(ty_name)
#             while src.last_token() == ',':
#                 src.process_token()
#                 ty_name = src.last_token()  # todo: parse type
#                 src.process_token()
#                 result_types.append(ty_name)
#             src.process_token(')')
#         else:
#             ty_name = src.last_token()  # todo: parse type
#             src.process_token()
#             result_types.append(ty_name)
#
#         def set_result_type(op: Op):
#             assert all(value.ty <= ty for ty, value in zip(result_types, op.results))
#             op.results = [td.Value(ty) for ty in result_types] if result_types else []
#
#         return set_result_type


class Op(serializable.TextSerializable):
    op_name: str = None
    op_type_dict: typing.Dict[str, typing.Type['Op']] = {}

    @staticmethod
    def register_op_cls(name: str, op_type: typing.Type['Op']):
        assert name not in Op.op_type_dict
        Op.op_type_dict[name] = op_type

    @staticmethod
    def get_op_cls(name: str):
        assert name in Op.op_type_dict
        return Op.op_type_dict[name]

    def __init_subclass__(cls):
        if cls.op_name is not None:
            Op.register_op_cls(cls.op_name, cls)

    def __init__(self, loc: location.Location, operands: typing.List[td.Value] = None,
                 result_types: typing.List[mlir_type.Type] = None, blocks: typing.List['Block'] = None):
        self.location = loc
        self.operands = operands if operands else []
        self.results = [td.Value(ty) for ty in result_types] if result_types else []
        self.blocks = blocks

    @classmethod
    def build(cls, loc: location.Location, operands: typing.List[td.Value] = None,
              result_types: typing.List[mlir_type.Type] = None, blocks: typing.List['Block'] = None):
        return cls(loc=loc, operands=operands, result_types=result_types, blocks=blocks)

    @classmethod
    def get_assembly_format(cls) -> formater.Format:
    #     return OpFormat(cls)
    #
    # def attr_dict_format(self):
    #     _ = self
    #     return AttrDictFormat()
    #
    # def operand_name_format(self, idx):
    #     return OperandNameFormat(self, idx)
    #
    # def operand_type_format(self, idx):
    #     return OperandTypeFormat(self, idx)
    #
    # def operands_format(self, detail=False, show_type=False):
    #     if detail:
    #         def print_func(dst: scoped_text_printer.ScopedTextPrinter):
    #             self.print_arguments_detail(dst, print_type=show_type)
    #     else:
    #         print_func = self.print_arguments
    #
    #     return print_func, NotImplemented
    #
    # def result_types_format(self):
    #     return self.print_result_types, NotImplemented

    def print(self, dst: scoped_text_printer.ScopedTextPrinter):
        if len(self.results) != 0:
            self.print_return_values(dst)
            dst.print(' = ', end='')
        dst.print(self.op_name, end='')

        self.print_assembly_format(dst)

        dst.print(dst.sep, end='')
        self.print_loc(dst)
        dst.print_newline()

    def print_assembly_format(self, dst: scoped_text_printer.ScopedTextPrinter):
        assembly_format = self.get_assembly_format()
        # assert isinstance(assembly_format, list)
        # for item in assembly_format:
        #     if isinstance(item, str):
        #         dst.print(item, end='')
        #     elif isinstance(item, serializable.TextSerializable):
        #         item.print(dst)
        #     elif isinstance(item, tuple) and len(item) == 2:
        #         # (printer, parser)
        #         assert isinstance(item[0], typing.Callable)
        #         item[0](dst)
        #     else:
        #         raise ValueError(f'Unknown assembly format item: {item}')

    def print_return_values(self, dst: scoped_text_printer.ScopedTextPrinter):
        for result_value in tools.with_sep(self.results, lambda: dst.print(',')):
            result_name = dst.next_unused_symbol('%')
            dst.insert_value_name(result_value, result_name)
            dst.print(result_name, end='')

    def print_arguments_detail(self, dst: scoped_text_printer.ScopedTextPrinter, print_type: bool = False):
        if len(self.operands) > 0:
            dst.print('(', end='')
            for operand_value in tools.with_sep(self.operands, lambda: dst.print(',')):
                operand_name = dst.lookup_value_name(operand_value)
                dst.print(operand_name, end='')
                if print_type:
                    dst.print(' :')
                    operand_value.ty.print(dst)
            dst.print(')', end='')

    def print_arguments(self, dst: scoped_text_printer.ScopedTextPrinter):
        for operand_value in tools.with_sep(self.operands, lambda: dst.print(',')):
            operand_name = dst.lookup_value_name(operand_value)
            dst.print(operand_name, end='')

    def print_result_types(self, dst: serializable.TextPrinter):
        if len(self.results) > 1:
            dst.print('(', end='')
            for result_value in tools.with_sep(self.results, lambda: dst.print(',')):
                result_value.ty.print(dst)
            dst.print(')', end='')
        elif len(self.results) == 1:
            self.results[0].ty.print(dst)
        else:
            dst.print('none', end='')

    def print_loc(self, dst: serializable.TextPrinter):
        self.location.print(dst)

    @classmethod
    def parse(cls, src: scoped_text_parser.ScopedTextParser):
        return_name = None
        if src.last_token() == '%':
            src.process_token()
            return_name = '%' + src.last_token()
            src.process_token(check_kind=serializable.TokenKind.Identifier)
            src.process_token('=')

        op_cls_name = src.last_token()
        src.process_token(check_kind=serializable.TokenKind.Identifier)
        op_cls = Op.get_op_cls(op_cls_name)
        value = op_cls.parse_assembly_format(src)

        src.define_var(return_name, value)

    @classmethod
    def parse_assembly_format(cls, src: scoped_text_parser.ScopedTextParser):
        assembly_format = cls.get_assembly_format()
        assert isinstance(assembly_format, list)
        for item in assembly_format:
            if isinstance(item, str):
                item = item.strip()
                src.process_token(item)
                return item
            elif isinstance(item, serializable.TextSerializable):
                return item.parse(src)
            elif isinstance(item, tuple) and len(item) == 2:
                # (printer, parser)
                assert isinstance(item[1], typing.Callable)
                return item[0](src)
            else:
                raise ValueError(f'Unknown assembly format item: {item}')


class Block(serializable.TextSerializable):
    def __init__(self, input_types=None):
        self.arguments = [td.Value(ty) for ty in input_types] if input_types else []
        self.op_list: typing.List[Op] = []

    def add_ops(self, op_list: typing.List[Op]):
        self.op_list.extend(op_list)

    def print(self, dst: scoped_text_printer.ScopedTextPrinter):
        for op in self.op_list:
            dst.print_ident()
            op.print(dst)


class ModuleOp(Op):
    op_name = 'module'

    def __init__(self, loc: location.Location, module_name: str, func_dict: typing.Dict[str, Op]):
        super().__init__(loc)
        self.module_name = module_name
        self.func_dict = func_dict

    def get_assembly_format(cls) -> typing.Optional[typing.List[typing.Any]]:
        assembly_format = super().get_assembly_format()
        assembly_format.append((cls.print_content, NotImplemented))
        return assembly_format

    def print_content(self, dst: scoped_text_printer.ScopedTextPrinter):
        dst.print('{', end='\n')
        with dst:
            for func in self.func_dict.values():
                dst.print_ident()
                func.print(dst)
        dst.print('}', end='')

    def dump(self):
        printer = scoped_text_printer.ScopedTextPrinter(file=sys.stdout)
        self.print(printer)
