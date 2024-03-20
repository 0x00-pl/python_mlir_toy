import sys
import typing

from python_mlir_toy.common import serializable, scoped, td


class ScopedTextParser(serializable.TextParser, scoped.Scoped):
    def __init__(self, file: typing.TextIO = sys.stdin, filename: str = 'unknown'):
        serializable.TextParser.__init__(self, file, filename)
        self.symbol_table = scoped.SymbolTable[td.Value]()
        scoped.Scoped.__init__(self, [self.symbol_table])

    def define_var(self, name: str, value: td.Value):
        self.symbol_table.insert(name, value)

    def lookup_var(self, name: str):
        return self.symbol_table.lookup(name)


# def parse_op(src: ScopedTextParser) -> serializable.TextSerializable:
#     assert src.last_token_kind() == serializable.TokenKind.Identifier
#     name_list = [src.last_token()]
#     src.process_token()
#
#     while src.last_token() == '.':
#         src.process_token()
#         assert src.last_token_kind() == serializable.TokenKind.Identifier
#         name_list.append(src.last_token())
#         src.process_token()
#
#     name = '.'.join(name_list)
#     assert name in op_type_dict
#     return op_type_dict[name]
