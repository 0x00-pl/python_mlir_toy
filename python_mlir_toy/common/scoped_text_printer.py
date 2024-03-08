import sys
import typing

from python_mlir_toy.common import serializable, symbol_table, td
from python_mlir_toy.common.indent import Indent


class ScopedTextPrinter(serializable.TextPrinter):
    def __init__(self, sep=' ', end=' ', file: typing.TextIO = sys.stdout):
        super().__init__(sep, end, file)
        self.indent = Indent()
        self.scope = symbol_table.SymbolTable()
        self.value_name_dict: typing.List[typing.Dict[td.Value, str]] = [{}]

    def print_ident(self):
        self.print(str(self.indent), end='')

    def print_escaped_str(self, string: str):
        escaped = string.replace('\\', '\\\\').replace('"', '\\"')
        self.print(f'"{escaped}"')

    def lookup_value_name(self, value: td.Value):
        for scope in reversed(self.value_name_dict):
            if value in scope:
                return scope[value]
        return None

    def next_unused_symbol(self, prefix: str = '%'):
        return self.scope.next_unused_symbol(prefix)

    def insert_value_name(self, value: td.Value, name: str):
        self.scope.insert(name, value)
        self.value_name_dict[-1][value] = name

    def __enter__(self):
        self.scope.__enter__()
        self.value_name_dict.append({})

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.scope.__exit__(exc_type, exc_val, exc_tb)
        self.value_name_dict.pop()
