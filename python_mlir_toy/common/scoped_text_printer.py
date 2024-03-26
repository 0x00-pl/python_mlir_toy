import sys
import typing

from python_mlir_toy.common import serializable, scoped, td


class ScopedTextPrinter(serializable.TextPrinter, scoped.Scoped):
    def __init__(self, sep=' ', end=' ', file: typing.TextIO = sys.stdout):
        serializable.TextPrinter.__init__(self, sep, end, file)
        self.indent = scoped.Indent()
        self.symbol_table_scope = scoped.SymbolTable[td.Value]()
        self.value_name_scope = scoped.KVScoped[td.Value, str]()
        scoped.Scoped.__init__(self, [self.indent, self.symbol_table_scope, self.value_name_scope])

    def print_ident(self):
        self.print(str(self.indent), end='')

    def print_escaped_str(self, string: str):
        escaped = string.replace('\\', '\\\\').replace('"', '\\"')
        self.print(f'"{escaped}"')

    def lookup_value_name(self, value: td.Value) -> typing.Optional[str]:
        return self.value_name_scope.lookup(value)

    def next_unused_symbol(self, prefix: str = '%'):
        return self.symbol_table_scope.next_unused_symbol(prefix)

    def insert_value_name(self, value: td.Value, name: str):
        self.symbol_table_scope.insert(name, value)
        self.value_name_scope.insert(value, name)

    def insert_value_and_generate_name(self, value: td.Value, prefix: str = '%') -> str:
        assert self.lookup_value_name(value) is None
        new_name = self.next_unused_symbol(prefix)
        self.insert_value_name(value, new_name)
        return new_name
