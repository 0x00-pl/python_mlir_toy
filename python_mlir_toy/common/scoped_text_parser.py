import sys
import typing

from python_mlir_toy.common import serializable, scoped, td

op_type_dict: typing.Dict[str, typing.Type[serializable.TextSerializable]] = {}


def register_op(name: str, op_type: typing.Type[serializable.TextSerializable]):
    assert name not in op_type_dict
    op_type_dict[name] = op_type


def register_cls(name: str):
    def inner(cls: typing.Type[serializable.TextSerializable]):
        register_op(name, cls)
        return cls
    return inner


def get_registered_op(name: str):
    assert name in op_type_dict
    return op_type_dict[name]


class ScopedTextParser(serializable.TextParser, scoped.Scoped):
    def __init__(self, file: typing.TextIO = sys.stdin, filename: str = 'unknown'):
        serializable.TextParser.__init__(self, file, filename)
        self.symbol_table = scoped.SymbolTable[td.Value]()
        self.value_name = scoped.KVScoped[td.Value, str]()
        scoped.Scoped.__init__(self, [self.symbol_table, self.value_name])

    def define_var(self, name: str, value: td.Value):
        self.symbol_table.insert(name, value)
        self.value_name.insert(value, name)


def parse_op(src: ScopedTextParser) -> serializable.TextSerializable:
    assert src.last_token_kind() == serializable.TokenKind.Identifier
    name_list = [src.last_token()]
    src.process_token()

    while src.last_token() == '.':
        src.process_token()
        assert src.last_token_kind() == serializable.TokenKind.Identifier
        name_list.append(src.last_token())
        src.process_token()

    name = '.'.join(name_list)
    assert name in op_type_dict
    return op_type_dict[name]
