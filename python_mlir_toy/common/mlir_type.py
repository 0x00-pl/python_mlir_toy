import typing


class Type:
    def __le__(self, other):
        return isinstance(other, type(self))

    def __eq__(self, other):
        return other <= self <= other


class IntType(Type):
    pass


class Float64Type(Type):
    pass


class TensorType(Type):
    def __init__(self, element_type: Type):
        self.element_type = element_type

    def __le__(self, other):
        return super().__le__(other) and self.element_type == other.element_type


class RankedTensorType(TensorType):
    def __init__(self, element_type: Type, shape: typing.List[int]):
        super().__init__(element_type)
        self.shape = shape

    def __le__(self, other):
        return super().__le__(other) and self.shape == other.shape


# class UnrankedTensorType(Tensor):
#     def __init__(self, element_type: Type):
#         super().__init__(element_type)


class FunctionType(Type):
    def __init__(self, inputs: typing.List[Type], outputs: typing.List[Type]):
        self.inputs = inputs
        self.outputs = outputs


def F64TensorType():
    return TensorType(Float64Type())


def RankedF64TensorType(shape: typing.List[int]):
    return RankedTensorType(Float64Type(), shape)


