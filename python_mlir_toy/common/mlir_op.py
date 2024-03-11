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
        self.print_return_values(dst)
        dst.print(self.name)
        self.print_arguments(dst)
        if len(self.results) > 0:
            dst.print(' : ', end='')
            self.print_results(dst)
        self.print_content(dst)
        dst.print()
        self.print_loc(dst)
        dst.print_newline()

    def print_return_values(self, dst: scoped_text_printer.ScopedTextPrinter):
        if len(self.results) != 0:
            for result_value in tools.with_sep(self.results, lambda: dst.print(',')):
                result_name = dst.next_unused_symbol('%')
                dst.insert_value_name(result_value, result_name)
                dst.print(result_name)

            dst.print('=')

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

    def print_results(self, dst: serializable.TextPrinter):
        if len(self.results) > 1:
            dst.print('(', end='')
            for result_value in tools.with_sep(self.results, lambda: dst.print(',')):
                result_value.ty.print(dst)
            dst.print(')', end='')
        elif len(self.results) == 1:
            self.results[0].ty.print(dst)
        else:
            dst.print('none', end='')

    def print_content(self, dst: serializable.TextPrinter):
        raise NotImplementedError(f'{self.name}:{type(self)} does not implement print_content()')

    def print_loc(self, dst: serializable.TextPrinter):
        self.location.print(dst)


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
    def __init__(self, loc: location.Location, module_name: str, func_dict: typing.Dict[str, Op]):
        super().__init__(loc, 'module')
        self.module_name = module_name
        self.func_dict = func_dict

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
