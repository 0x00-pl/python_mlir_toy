import typing


class Type:
    def __le__(self, other):
        return isinstance(other, type(self))

    def __eq__(self, other):
        return other <= self <= other


class NoneType(Type):
    pass


class OpaqueType(Type):
    def __init__(self, dialect: str, type_name: str):
        self.dialect = dialect
        self.type_name = type_name


class IndexType(Type):
    pass


class IntegerType(Type):
    def __init__(self, bits: int, signed: bool):
        self.bits = bits
        self.signed = signed

    def __le__(self, other):
        return super().__le__(other) and self.bits == other.bits and self.signed == other.signed


class Float32Type(Type):
    pass


class Float64Type(Type):
    pass


class ComplexType(Type):
    def __init__(self, element_type: Type):
        self.element_type = element_type

    def __le__(self, other):
        return super().__le__(other) and self.element_type == other.element_type


class VectorType(Type):
    def __init__(self, element_type: Type, dims: typing.List[int], is_scalable_dims: typing.List[bool] = None):
        self.element_type = element_type
        self.dims = dims
        self.is_scalable_dims = is_scalable_dims

    def __le__(self, other):
        if self.is_scalable_dims is not None or other.is_scalable_dims is not None:
            return False
        return super().__le__(other) and self.element_type == other.element_type and self.dims == other.dims


class TensorType(Type):
    def __init__(self, element_type: Type):
        self.element_type = element_type

    def __le__(self, other):
        return super().__le__(other) and self.element_type == other.element_type


class RankedTensorType(TensorType):
    def __init__(self, element_type: Type, shape: typing.List[int]):
        super().__init__(element_type)
        self.shape = shape

    def is_dynamic_dim(self, idx):
        return self.shape[idx] < 0

    def __le__(self, other):
        return super().__le__(other) and self.shape == other.shape


class TupleType(Type):
    def __init__(self, types: typing.List[Type]):
        self.types = types

    def __le__(self, other):
        return super().__le__(other) and all(t1 <= t2 for t1, t2 in zip(self.types, other.types))


class FunctionType(Type):
    def __init__(self, inputs: typing.List[Type], outputs: typing.List[Type]):
        self.inputs = inputs
        self.outputs = outputs


def F64TensorType():
    return TensorType(Float64Type())


def RankedF64TensorType(shape: typing.List[int]):
    return RankedTensorType(Float64Type(), shape)
