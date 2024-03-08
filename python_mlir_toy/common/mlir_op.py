import sys
import typing

from python_mlir_toy.common import location, td, serializable, scoped_text_printer, mlir_type, tools


class Op(serializable.TextSerializable):
    def __init__(self, loc: location.Location, name: str, operands: typing.List[td.Value] = None,
                 result_types: typing.List[mlir_type.Type] = None, blocks: typing.List['Block'] = None):
        self.location = loc
        self.name = name
        self.operands = operands if operands else []
        self.results = [td.Value(ty) for ty in result_types] if result_types else []
        self.blocks = blocks

    def print(self, dst: scoped_text_printer.ScopedTextPrinter):
        if len(self.results) != 0:
            for result_value in tools.with_sep(self.results, lambda: dst.print(',')):
                result_name = dst.next_unused_symbol('%')
                dst.insert_value_name(result_value, result_name)
                dst.print(result_name)

            dst.print('=')

        dst.print(self.name)

        self.print_arguments(dst)
        self.print_content(dst)
        dst.print_newline()

    def print_arguments(self, dst: scoped_text_printer.ScopedTextPrinter):
        if len(self.operands) > 0:
            dst.print('(', end='')
            for operand_value in tools.with_sep(self.operands, lambda: dst.print(',')):
                operand_name = dst.lookup_value_name(operand_value)
                dst.print(operand_name)
                dst.print(':')
                operand_value.ty.print(dst)
            dst.print(')')

    def print_content(self, dst: serializable.TextPrinter):
        raise NotImplementedError(f'{self.name}:{type(self)} does not implement print_content()')


class Block(serializable.TextSerializable):
    def __init__(self, input_types=None):
        self.arguments = [td.Value(ty) for ty in input_types] if input_types else []
        self.op_list: typing.List[Op] = []

    def add_ops(self, op_list: typing.List[Op]):
        self.op_list.extend(op_list)

    def print(self, dst: scoped_text_printer.ScopedTextPrinter):
        dst.print('{', end='\n')
        for op in self.op_list:
            dst.print_ident()
            op.print(dst)
        dst.print('}', end='\n')


class ModuleOp(Op):
    def __init__(self, loc: location.Location, module_name: str, func_dict: typing.Dict[str, Op]):
        super().__init__(loc, 'module')
        self.module_name = module_name
        self.func_dict = func_dict

    def print_content(self, dst: scoped_text_printer.ScopedTextPrinter):
        dst.print('{', end='\n')
        with dst.indent:
            for func in self.func_dict.values():
                dst.print_ident()
                func.print(dst)
        dst.print('}', end='\n')

    def dump(self):
        printer = scoped_text_printer.ScopedTextPrinter(file=sys.stdout)
        self.print(printer)
