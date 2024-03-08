from python_mlir_toy.common import serializable
from python_mlir_toy.common.serializable import TextPrinter


class Location(serializable.TextSerializable):
    pass


class UnknownLocation(Location):
    def print(self, dst: TextPrinter):
        dst.print('loc(unknown)')


class FileLineColLocation(Location):
    def __init__(self, filename, line, column):
        self.filename = filename
        self.line = line
        self.column = column

    def print(self, dst: TextPrinter):
        dst.print(f'loc({self.filename}:{self.line}:{self.column})')
