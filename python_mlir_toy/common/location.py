from python_mlir_toy.common import serializable
from python_mlir_toy.common.serializable import TextPrinter


class Location(serializable.TextSerializable):
    pass


class UnknownLocation(Location):
    def print(self, dst: TextPrinter):
        dst.print('loc(unknown)', end='')


class FileLineColLocation(Location):
    def __init__(self, filename, line, column):
        self.filename = filename
        self.line = line
        self.column = column

    def print(self, dst: TextPrinter):
        dst.print(f'loc("{self.filename}":{self.line}:{self.column})', end='')


def parse_location(src: serializable.TextParser):
    if src.last_token() != 'loc':
        return None

    src.drop_token()
    src.drop_token('(')
    if src.last_token() == 'unknown':
        ret = UnknownLocation()
        src.drop_token()
    elif src.last_token_kind() == serializable.TokenKind.String:
        filename = src.last_token()
        src.drop_token()
        src.drop_token(':')
        line = src.last_token()
        src.drop_token()
        src.drop_token(':')
        column = src.last_token()
        src.drop_token()
        ret = FileLineColLocation(filename, line, column)
    else:
        raise ValueError('Unknown location format')

    return ret
