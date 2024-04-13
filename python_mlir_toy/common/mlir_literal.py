import typing

from python_mlir_toy.common import serializable, tools, mlir_type


class Literal(serializable.TextSerializable):
    name = None
    type_dict: typing.Dict[str, typing.Type['Literal']] = {}

    def __init_subclass__(cls, **kwargs):
        if cls.name is not None:
            Literal.type_dict[cls.name] = cls

    def get_type(self):
        assert self.name is None
        return mlir_type.NoneType()

    def print(self, dst: serializable.TextPrinter):
        if self.name is not None:
            dst.print(self.name, end='')
        else:
            dst.print('unknown_type', end='')

    @classmethod
    def parse(cls, src: serializable.TextParser):
        src.drop_token(cls.name)
        return cls()


class FloatLiteral(Literal):
    def __init__(self, value: float):
        self.value = value

    def get_type(self):
        return mlir_type.Float64Type()

    def print(self, dst: serializable.TextPrinter):
        dst.print(self.value, end='')

    @classmethod
    def parse(cls, src: serializable.TextParser):
        value = src.last_token()
        src.drop_token()
        return cls(float(value))


class TensorLiteral(Literal):
    name = 'tensor'

    def __init__(self, shape: typing.List[int], values: typing.List[float]):
        self.shape = shape
        self.values = values

    def get_type(self):
        return mlir_type.RankedF64TensorType(self.shape)

    def print(self, dst: serializable.TextPrinter):
        dst.print(f'{self.name}<[', end='')
        for value in tools.with_sep(self.values, lambda: dst.print(',')):
            dst.print(value, end='')
        dst.print(']> : ', end='')
        ty = mlir_type.RankedF64TensorType(self.shape)
        ty.print(dst)
        dst.print()

    @classmethod
    def parse(cls, src: serializable.TextParser):
        src.drop_token(cls.name)
        src.drop_token('<')
        src.drop_token('[')
        values = []
        while src.last_token() != ']':
            assert src.last_token_kind() == serializable.TokenKind.Number
            values.append(float(src.last_token()))
            src.drop_token()
            if src.last_token() == ',':
                src.drop_token()
        src.drop_token(']')
        src.drop_token('>')
        src.drop_token(':')
        ty = mlir_type.parse_type(src)
        assert isinstance(ty, mlir_type.RankedTensorType)
        return cls(ty.shape, values)


class DenseTensorLiteral(TensorLiteral):
    name = 'dense'


def parse_literal(src: serializable.TextParser):
    token = src.last_token()
    if isinstance(str, (int, float)):
        return FloatLiteral.parse(src)
    elif token == 'unknown_type':
        src.drop_token()
        return None
    elif token in Literal.type_dict:
        return Literal.type_dict[token].parse(src)
    else:
        raise NotImplementedError(f'Unsupported literal type: {token}')
