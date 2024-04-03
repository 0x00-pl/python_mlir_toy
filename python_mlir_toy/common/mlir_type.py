import typing

from python_mlir_toy.common import serializable, tools
from python_mlir_toy.common.serializable import TextPrinter, TextParser


class Type(serializable.TextSerializable):
    name = None
    type_dict: typing.Dict[str, typing.Type['Type']] = {}

    def __init_subclass__(cls, **kwargs):
        if cls.name is not None:
            Type.type_dict[cls.name] = cls

    def __le__(self, other):
        return isinstance(other, type(self))

    def __eq__(self, other):
        return other <= self <= other

    def print(self, dst: TextPrinter):
        if self.name is not None:
            dst.print(self.name, end='')
        else:
            dst.print('unknown_type', end='')

    @classmethod
    def parse(cls, src: TextParser):
        src.drop_token(cls.name)
        return cls()


class NoneType(Type):
    name = 'none'


def print_dialect_symbol(dst: TextPrinter, prefix: str, dialect_name: str, symbol_name: str):
    dst.print(f'{prefix}{dialect_name}.{symbol_name}')


class OpaqueType(Type):
    def __init__(self, dialect: str, type_name: str):
        self.dialect = dialect
        self.type_name = type_name

    def print(self, dst: TextPrinter):
        print_dialect_symbol(dst, '!', self.dialect, self.type_name)


class IndexType(Type):
    name = 'index'


class IntType(Type):
    def __init__(self, bits: int, signed: bool):
        self.bits = bits
        self.signed = signed

    def __le__(self, other):
        return super().__le__(other) and self.bits == other.bits and self.signed == other.signed

    def print(self, dst: TextPrinter):
        dst.print(f'{"s" if self.signed else "u"}{self.bits}i', end='')

    @classmethod
    def parse(cls, src: TextParser):
        assert src.last_token_kind() == serializable.TokenKind.Identifier
        type_str: str = src.last_token()
        src.drop_token()
        signed = not type_str.startswith('u')
        bits = int(type_str.strip('sui'))
        return cls(bits=bits, signed=signed)


class Float32Type(Type):
    name = 'f32'


class Float64Type(Type):
    name = 'f64'


class ComplexType(Type):
    name = 'complex'

    def __init__(self, element_type: Type):
        self.element_type = element_type

    def __le__(self, other):
        return super().__le__(other) and self.element_type == other.element_type

    def print(self, dst: TextPrinter):
        dst.print(f'complex<', end='')
        self.element_type.print(dst)
        dst.print('>', end='')

    @classmethod
    def parse(cls, src: TextParser):
        src.drop_token('complex')
        src.drop_token('<')
        element_type = parse_type(src)
        src.drop_token('>')
        return cls(element_type)


class VectorType(Type):
    name = 'vector'

    def __init__(self, element_type: Type, dims: typing.List[int], is_scalable_dims: typing.List[bool] = None):
        self.element_type = element_type
        self.dims = dims
        self.is_scalable_dims = is_scalable_dims

    def __le__(self, other):
        if self.is_scalable_dims is not None or other.is_scalable_dims is not None:
            return False
        return super().__le__(other) and self.element_type == other.element_type and self.dims == other.dims

    def print(self, dst: TextPrinter):
        dst.print('vector<', end='')
        for idx, dim in enumerate(tools.with_sep(self.dims, lambda: dst.print(','))):
            if self.is_scalable_dims is not None and self.is_scalable_dims[idx]:
                dst.print(f'[{dim}]', end='')
            else:
                dst.print(dim)
        dst.print(f'x', end='')
        self.element_type.print(dst)
        dst.print('>', end='')

    @classmethod
    def parse(cls, src: TextParser):
        src.drop_token('vector')
        src.drop_space()
        dims = []
        is_scalable_dims = []
        element_type = None
        src.process_char('<')
        while src.last_char() != '>':
            src.process_char()
            if src.last_char() == '[':
                src.process_char()
                num_str = ''
                while src.last_char().isdigit():
                    num_str += src.last_char()
                    src.process_char()
                src.process_char(']')
                dims.append(int(num_str))
                is_scalable_dims.append(True)
            elif src.last_char().isdigit():
                num_str = ''
                while src.last_char().isdigit():
                    num_str += src.last_char()
                    src.process_char()
                dims.append(int(num_str))
                is_scalable_dims.append(False)
            else:
                element_type = parse_type(src)
                break
        src.drop_token('>')
        return cls(element_type, dims, is_scalable_dims)


class TensorType(Type):
    def __init__(self, element_type: Type):
        self.element_type = element_type

    def __le__(self, other):
        return super().__le__(other) and self.element_type == other.element_type

    def print(self, dst: TextPrinter):
        dst.print(f'tensor<*x', end='')
        self.element_type.print(dst)
        dst.print('>', end='')


class RankedTensorType(TensorType):
    name = 'tensor'

    def __init__(self, element_type: Type, shape: typing.List[int]):
        super().__init__(element_type)
        self.shape = shape

    def is_dynamic_dim(self, idx):
        return self.shape[idx] < 0

    def __le__(self, other):
        return super().__le__(other) and self.shape == other.shape

    def print(self, dst: TextPrinter):
        dst.print('tensor<', end='')
        for dim in tools.with_sep(self.shape, lambda: dst.print('x', end='')):
            dst.print('?' if dim < 0 else str(dim), end='')
        dst.print(f'x', end='')
        self.element_type.print(dst)
        dst.print('>', end='')

    @classmethod
    def parse(cls, src: TextParser):
        assert src.last_token() == 'tensor'
        src.drop_space()
        src.drop_char('<')
        if src.cur_char() == '*':
            src.drop_char()
            src.drop_char('x')
            src.drop_token()
            element_type = parse_type(src)
            src.drop_token('>')
            return TensorType(element_type)

        shape = []
        element_type = None
        while src.last_char() != '>':
            if src.last_char() == 'x':
                src.process_char()
            dim_str = ''
            if src.last_char().isdigit():
                while src.last_char().isdigit():
                    dim_str += src.last_char()
                    src.process_char()
                shape.append(int(dim_str))
            else:
                element_type = parse_type(src)
                assert src.last_token() == '>'
        src.drop_token('>')
        assert element_type is not None
        return cls(element_type, shape)


class TupleType(Type):
    name = 'tuple'

    def __init__(self, types: typing.List[Type]):
        self.types = types

    def __le__(self, other):
        return super().__le__(other) and all(t1 <= t2 for t1, t2 in zip(self.types, other.types))

    def print(self, dst: TextPrinter):
        dst.print('tuple<')
        for input_ty in tools.with_sep(self.types, lambda: dst.print(', ')):
            input_ty.print(dst)
        dst.print('>')

    @classmethod
    def parse(cls, src: TextParser):
        src.drop_token('tuple')
        src.drop_space()
        src.drop_token('<')
        src.drop_token(')')
        types = []
        while src.last_char() != ')':
            if src.last_char() == ',':
                src.process_char()
                src.drop_space()
            sub_type = parse_type(src)
            types.append(sub_type)
        src.drop_token(')')
        src.drop_token('>')
        return cls(types)


class FunctionType(Type):
    def __init__(self, inputs: typing.List[Type], outputs: typing.List[Type]):
        self.inputs = inputs
        self.outputs = outputs

    def print(self, dst: TextPrinter):
        dst.print('(', end='')
        for input_ty in tools.with_sep(self.inputs, lambda: dst.print(',')):
            input_ty.print(dst)
        dst.print(') ->')

        if len(self.outputs) == 1:
            self.outputs[0].print(dst)
            dst.print()
        else:
            dst.print('(', end='')
            for output_ty in tools.with_sep(self.outputs, lambda: dst.print(',')):
                output_ty.print(dst)
            dst.print(')')


def print_type_list(dst: TextPrinter, type_list: typing.List[Type], sep: str = ', '):
    for result_type in tools.with_sep(type_list, lambda: dst.print(sep)):
        result_type.print(dst)


def parse_type_list(src: TextParser, sep: str = ','):
    if src.last_token() == '(':
        src.drop_token()
        type_list = [parse_type(src)]
        while src.last_token() != ')':
            if src.last_token() == sep:
                src.drop_token()
            type_list.append(parse_type(src))
        src.drop_token(')')
    else:
        type_list = [parse_type(src)]
    return type_list


def parse_type(src: TextParser):
    src.drop_space()
    type_name: str = src.last_token()
    if type_name == 'unknown_type':
        src.drop_token()
        return None
    elif type_name[0] in 'sui' and 'i' in type_name and type_name.lstrip('sui').isdigit():
        src.drop_token()
        return IntType(int(type_name.lstrip('sui')), type_name[0] != 'u')
    elif type_name == '(':
        # function type
        arg_types = parse_type_list(src)
        src.drop_token('-')
        src.drop_token('>')
        result_types = parse_type_list(src)
        return FunctionType(arg_types, result_types)
    else:
        assert type_name in Type.type_dict
        cls = Type.type_dict[type_name]
        return cls.parse(src)


def parse_function_type(src: TextParser):
    arg_types = parse_type_list(src)
    src.drop_token('-')
    src.drop_token('>')
    result_types = parse_type_list(src)
    return FunctionType(arg_types, result_types)


def F64TensorType():
    return TensorType(Float64Type())


def RankedF64TensorType(shape: typing.List[int]):
    return RankedTensorType(Float64Type(), shape)
