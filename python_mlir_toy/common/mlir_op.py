import typing

from python_mlir_toy.common import location, td, serializable, scoped_text_printer, mlir_type, scoped_text_parser, \
    formater, tools


class TypeFormat(formater.Format):
    def print(self, obj, dst: serializable.TextPrinter):
        assert isinstance(obj, mlir_type.Type)
        obj.print(dst)

    def parse(self, src: serializable.TextParser):
        return mlir_type.parse_type(src)


class TypeListFormat(formater.Format):
    def __init__(self, parentheses_required: bool, prefix: str | formater.ConstantStrFormat = ' : '):
        self.types_format = formater.RepeatFormat(TypeFormat(), ', ')
        self.prefix_format = formater.ConstantStrFormat(prefix) if isinstance(prefix, str) else prefix
        self.parentheses_required = parentheses_required

    def print(self, obj, dst: serializable.TextPrinter):
        if obj is None:
            return

        assert isinstance(obj, list)
        self.prefix_format.print(serializable.empty, dst)
        if len(obj) != 1 or self.parentheses_required:
            dst.print('(')
            self.types_format.print(obj, dst)
            dst.print(')')
        else:
            self.types_format.print(obj, dst)

    def parse(self, src: serializable.TextParser) -> typing.List[typing.Any]:
        if src.last_token() != self.prefix_format.text.strip():
            return []

        self.prefix_format.parse(src)
        if src.last_token() == '(':
            src.drop_token()
            ret = self.types_format.parse(src)
            src.drop_token(')')
        else:
            ret = self.types_format.parse(src)
        return ret


#
#
# class OpFormat(formater.Format):
#     def __init__(self, op_cls: typing.Type['Op']):
#         self.operands_format = formater.RepeatFormat(formater.VariableNameFormat('%'), ', ')
#         self.results_ty_format = formater.OptionalFormat(
#             formater.ListFormat([formater.ConstantStrFormat(' : '), TypeListFormat(parentheses_required=False)]),
#             (lambda result_type_list: len(result_type_list) > 0), ':')
#         self.op_cls = op_cls
#
#     def print(self, obj, dst: scoped_text_printer.ScopedTextPrinter):
#         operand_names = (dst.lookup_value_name(item) for item in obj.operands)
#         self.operands_format.print(operand_names, dst)
#         result_type_list = list(item.ty for item in obj.results)
#         self.results_ty_format.print(result_type_list, dst)
#
#     def parse(self, src: scoped_text_parser.ScopedTextParser):
#         loc = location.FileLineColLocation(*src.get_location())
#         operand_names = self.operands_format.parse(src)
#         operands = [src.lookup_var(operand_name) for operand_name in operand_names]
#         result_types = self.results_ty_format.parse(src)
#         return self.op_cls(loc, operands, result_types)


class Op(serializable.TextSerializable):
    op_name: str = None
    op_type_dict: typing.Dict[str, typing.Type['Op']] = {}

    _literal_format = formater.LiteralFormat()
    _location_format = formater.LocationFormat()
    _variable_name_format = formater.VariableNameFormat('%')
    _function_name_format = formater.VariableNameFormat('@')
    _results_name_format = formater.RepeatFormat(_variable_name_format, ', ')
    _op_name_format = formater.NamespacedSymbolFormat()
    _operands_format = formater.RepeatFormat(formater.VariableNameFormat('%'), ', ')
    _results_ty_format = TypeListFormat(parentheses_required=False)

    @staticmethod
    def register_op_cls(name: str, op_type: typing.Type['Op']):
        assert name not in Op.op_type_dict
        Op.op_type_dict[name] = op_type

    @staticmethod
    def get_op_cls(op_name: str):
        assert op_name in Op.op_type_dict
        return Op.op_type_dict[op_name]

    def __init_subclass__(cls):
        if cls.op_name is not None:
            Op.register_op_cls(cls.op_name, cls)

    def __init__(self, loc: location.Location, operands: typing.List[td.Value] = None,
                 result_types: typing.List[mlir_type.Type] = None, blocks: typing.List['Block'] = None):
        self.location = loc
        self.operands = operands if operands else []
        self.results = [td.Value(ty) for ty in result_types] if result_types else []
        self.blocks = blocks if blocks else []

    @classmethod
    def build_as_generic_op(cls, loc: location.Location, operands: typing.List[td.Value] = None,
                            result_types: typing.List[mlir_type.Type] = None, blocks: typing.List['Block'] = None):
        return cls(loc, operands, result_types, blocks)

    @classmethod
    def get_assembly_format(cls) -> formater.Format:
        def _print_op(obj, dst: scoped_text_printer.ScopedTextPrinter):
            operand_names = (dst.lookup_value_name(item) for item in obj.operands)
            cls._operands_format.print(operand_names, dst)
            result_type_list = list(item.ty for item in obj.results)
            cls._results_ty_format.print(result_type_list, dst)

        def _parse_op(src: scoped_text_parser.ScopedTextParser):
            operand_names = cls._operands_format.parse(src)
            operands = [src.lookup_var(operand_name) for operand_name in operand_names]
            result_types = cls._results_ty_format.parse(src)
            loc = cls._location_format.parse(src)
            return cls.build_as_generic_op(loc, operands, result_types)

        return formater.CustomFormat(_print_op, _parse_op)

    def print(self, dst: scoped_text_printer.ScopedTextPrinter):
        self.get_assembly_format().print(self, dst)

    @classmethod
    def parse(cls, src: scoped_text_parser.ScopedTextParser) -> 'Op':
        return cls.get_assembly_format().parse(src)


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


class FuncOp(Op):
    op_name = 'func'

    def __init__(self, loc: location.Location, function_name: str, arg_name_list: typing.List[str],
                 arg_loc_list: typing.List[location.Location | None] | None, function_type: mlir_type.FunctionType,
                 block: Block):
        super().__init__(loc, blocks=[block])
        self.function_name = function_name
        self.function_type = function_type
        self.arg_name_list = arg_name_list
        self.arg_loc_list = arg_loc_list if arg_loc_list is not None else [None] * len(arg_name_list)
        assert len(self.arg_name_list) == len(self.arg_loc_list)
        assert len(self.arg_name_list) == len(self.function_type.inputs)

    def print(self, dst: scoped_text_printer.ScopedTextPrinter):
        with dst:
            dst.print(self.function_name, '(', sep='', end='')
            for arg_name, arg_loc, arg_ty in tools.with_sep(
                    zip(self.arg_name_list, self.arg_loc_list, self.function_type.inputs), (lambda: dst.print(', '))):
                dst.insert_value_name(arg_ty, arg_name)
                dst.print(arg_name, ': ', sep='', end='')
                arg_ty.print(dst)
                if arg_loc is not None:
                    dst.print(' ')
                    arg_loc.print(dst)
            dst.print(')')
            if len(self.function_type.outputs) != 0:
                dst.print(' -> ')
                if len(self.function_type.outputs) == 1:
                    self.function_type.outputs[0].print(dst)
                else:
                    dst.print('(', end='')
                    mlir_type.print_type_list(dst, self.function_type.outputs)
                    dst.print(')')

            dst.print('{', end='\n')
            for op in self.blocks[0].op_list:
                result_names = [dst.insert_value_and_generate_name(item) for item in op.results]
                Op._results_name_format.print(result_names, dst)
                dst.print_ident()
                dst.print(' = ', end='')
                dst.print(op.op_name)
                op.print(dst)
                dst.print(end='\n')
            dst.print('}', end='\n')

    @classmethod
    def parse(cls, src: scoped_text_parser.ScopedTextParser) -> 'FuncOp':
        with src:
            function_name = Op._function_name_format.parse(src)
            arg_name_list = []
            arg_loc_list = []
            arg_ty_list = []
            output_ty_list = []
            src.drop_token('(')
            while src.last_token() != ')':
                if src.last_token() == ',':
                    src.drop_token()
                arg_name = Op._variable_name_format.parse(src)
                src.drop_token(':')
                arg_ty = mlir_type.parse_type(src)
                if src.last_token() == 'loc':
                    loc = Op._location_format.parse(src)
                else:
                    loc = None

                arg_name_list.append(arg_name)
                arg_loc_list.append(loc)
                arg_ty_list.append(arg_ty)
            src.drop_token(')')
            if src.last_token() == '-':
                src.drop_token('-')
                src.drop_token('>')
                output_ty_list = mlir_type.parse_type_list(src)

            block = Block(arg_ty_list)
            for name, value in zip(arg_name_list, block.arguments):
                src.define_var(name, value)

            function_type = mlir_type.FunctionType(arg_ty_list, output_ty_list)

            src.drop_token('{')
            while src.last_token() != '}':
                if src.last_token() == '%':
                    op_result_names = Op._results_name_format.parse(src)
                    src.drop_token('=')
                else:
                    op_result_names = []

                op_name = Op._op_name_format.parse(src)
                op_cls = Op.get_op_cls(op_name)
                op = op_cls.parse(src)
                block.op_list.append(op)

                for name, value in zip(op_result_names, op.results):
                    src.define_var(name, value)

            src.drop_token('}')
            loc = cls._location_format.parse(src)
            return FuncOp(loc, function_name, arg_name_list, arg_loc_list, function_type, block)


class ModuleOp(Op):
    op_name = 'module'

    def __init__(self, loc: location.Location, module_name: str, func_dict: typing.Dict[str, Op]):
        super().__init__(loc)
        self.module_name = module_name
        self.func_dict = func_dict

    def print(self, dst: scoped_text_printer.ScopedTextPrinter):
        with dst:
            dst.print('{', end='\n')
            for func in self.func_dict.values():
                dst.print_ident()
                func.print(dst)
            dst.print('}', end='')

    @classmethod
    def parse(cls, src: scoped_text_parser.ScopedTextParser) -> 'ModuleOp':
        func_dict = {}
        with src:
            src.drop_token('{')
            while src.last_token() != '}':
                func_op_name = Op._op_name_format.parse(src)
                func_cls = Op.get_op_cls(func_op_name)
                func_op = func_cls.parse(src)
                assert isinstance(func_op, FuncOp)
                func_dict[func_op.function_name] = func_op
            src.drop_token('}')
        return ModuleOp(src.last_location(), 'no_name', func_dict)


def parse_module(src: scoped_text_parser.ScopedTextParser):
    op_name = Op._op_name_format.parse(src)
    assert op_name == 'module'
    return ModuleOp.parse(src)
