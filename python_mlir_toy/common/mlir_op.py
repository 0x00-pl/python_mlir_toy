import sys
import typing

from python_mlir_toy.common import location, td, serializable, scoped_text_printer, mlir_type, tools, scoped_text_parser, formater


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
    def __init__(self, op_cls: typing.Type['Op']):
        self.operands_format = formater.RepeatFormat(formater.VariableNameFormat('%'), ', ')
        self.results_ty_format = formater.OptionalFormat(
            formater.ListFormat([formater.ConstantStrFormat(' : '), TypeListFormat(parentheses_required=False)]),
            (lambda result_type_list: len(result_type_list) > 0),
            ':'
        )
        self.op_cls = op_cls

    def print(self, obj, dst: scoped_text_printer.ScopedTextPrinter):
        operand_names = (dst.lookup_value_name(item) for item in obj.operands)
        self.operands_format.print(operand_names, dst)
        result_type_list = list(item.ty for item in obj.results)
        self.results_ty_format.print(result_type_list, dst)

    def parse(self, src: scoped_text_parser.ScopedTextParser):
        loc = location.FileLineColLocation(*src.last_location())
        operand_names = self.operands_format.parse(src)
        operands = [src.lookup_var(operand_name) for operand_name in operand_names]
        result_types = self.results_ty_format.parse(src)
        return self.op_cls(loc, operands, result_types)


class Op(serializable.TextSerializable):
    op_name: str = None
    op_type_dict: typing.Dict[str, typing.Type['Op']] = {}

    _results_name_format = formater.RepeatFormat(formater.VariableNameFormat('%'), ', ')
    _op_name_format = formater.NamespacedSymbolFormat()

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
    def get_assembly_format(cls) -> formater.Format:
        return OpFormat(cls)

    def print(self, dst: scoped_text_printer.ScopedTextPrinter):
        result_names = list(dst.insert_value_and_generate_name(item) for item in self.results)
        Op._results_name_format.print(result_names, dst)
        Op._op_name_format.print(self.op_name, dst)
        self.print_assembly_format(dst)

    def print_assembly_format(self, dst: scoped_text_printer.ScopedTextPrinter):
        assembly_format = self.get_assembly_format()
        assembly_format.print(self, dst)

    @classmethod
    def parse(cls, src: scoped_text_parser.ScopedTextParser) -> 'Op':
        result_names = Op._results_name_format.parse(src)
        op_name = Op._op_name_format.parse(src)
        op_cls = Op.get_op_cls(op_name)
        new_op = op_cls.parse_assembly_format(src)
        for name, value in zip(result_names, new_op.results):
            src.define_var(name, value)
        return new_op

    @classmethod
    def parse_assembly_format(cls, src: scoped_text_parser.ScopedTextParser):
        assembly_format = cls.get_assembly_format()
        return assembly_format.parse(src)


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


class FuncOp(Op, td.IsolatedFromAbove):
    op_name = 'toy.func'

    def __init__(self, loc: location.Location, function_name: str,
                 arg_name_list: typing.List[typing.Tuple[str, location.Location] | str], function_type: mlir_type.FunctionType,
                 block: Block):
        super().__init__(loc, blocks=[block])
        self.function_name = function_name
        self.function_type = function_type
        self.arg_name_list = arg_name_list
        assert len(arg_name_list) == len(function_type.inputs)


class ModuleFormat(OpFormat):
    def __init__(self):
        super().__init__(ModuleOp)
        self.function_arguments_format = formater.DictFormat(
            formater.VariableNameFormat('%'),
            formater.TypeFormat()
        )

    def print(self, obj, dst: scoped_text_printer.ScopedTextPrinter):
        assert isinstance(obj, ModuleOp)
        with dst:
            dst.print('{', end='\n')
            for operator in obj.blocks:
                operator.print(dst)
            dst.print('}', end='\n')

    def parse(self, src: scoped_text_parser.ScopedTextParser):
        loc = location.FileLineColLocation(*src.last_location())
        with src:
            op_list: typing.List[FuncOp] = []
            src.process_token('{')
            while src.last_token() != '}':
                op = Op.parse(src)
                assert isinstance(op, FuncOp)
                op_list.append(op)

            return ModuleOp(loc, 'no_name', {item.function_name: item for item in op_list})


class ModuleOp(Op):
    op_name = 'module'

    def __init__(self, loc: location.Location, module_name: str, func_dict: typing.Dict[str, Op]):
        super().__init__(loc)
        self.module_name = module_name
        self.func_dict = func_dict

    @classmethod
    def get_assembly_format(cls) -> formater.Format:
        return ModuleFormat()

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
