import io
import typing

from python_mlir_toy.common import serializable, tools
from python_mlir_toy.common.serializable import TextPrinter


class Type(serializable.TextSerializable):
    def __le__(self, other):
        return isinstance(other, type(self))

    def __eq__(self, other):
        return other <= self <= other


class NoneType(Type):
    def print(self, dst: TextPrinter):
        dst.print('none', end='')


def print_dialect_symbol(dst: TextPrinter, prefix: str, dialect_name: str, symbol_name: str):
    dst.print(f'{prefix}{dialect_name}.{symbol_name}')


class OpaqueType(Type):
    def __init__(self, dialect: str, type_name: str):
        self.dialect = dialect
        self.type_name = type_name

    def print(self, dst: TextPrinter):
        print_dialect_symbol(dst, '!', self.dialect, self.type_name)


class IndexType(Type):
    def print(self, dst: TextPrinter):
        dst.print('index', end='')


class IntegerType(Type):
    def __init__(self, bits: int, signed: bool):
        self.bits = bits
        self.signed = signed

    def __le__(self, other):
        return super().__le__(other) and self.bits == other.bits and self.signed == other.signed

    def print(self, dst: TextPrinter):
        dst.print(f'{"s" if self.signed else "u"}{self.bits}i', end='')


class Float32Type(Type):
    def print(self, dst: TextPrinter):
        dst.print('f32', end='')


class Float64Type(Type):
    def print(self, dst: TextPrinter):
        dst.print('f64', end='')


class ComplexType(Type):
    def __init__(self, element_type: Type):
        self.element_type = element_type

    def __le__(self, other):
        return super().__le__(other) and self.element_type == other.element_type

    def print(self, dst: TextPrinter):
        dst.print(f'complex<', end='')
        self.element_type.print(dst)
        dst.print('>', end='')


class VectorType(Type):
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


class TupleType(Type):
    def __init__(self, types: typing.List[Type]):
        self.types = types

    def __le__(self, other):
        return super().__le__(other) and all(t1 <= t2 for t1, t2 in zip(self.types, other.types))

    def print(self, dst: TextPrinter):
        dst.print('tuple<')
        for input_ty in tools.with_sep(self.types, lambda: dst.print(', ')):
            input_ty.print(dst)
        dst.print('>')


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


def F64TensorType():
    return TensorType(Float64Type())


def RankedF64TensorType(shape: typing.List[int]):
    return RankedTensorType(Float64Type(), shape)
