import typing


class Type:
    pass


class Float64Type(Type):
    pass


class UnrankedTensorType(Type):
    def __init__(self, element_type: Type):
        self.element_type = element_type


class FunctionType(Type):
    def __init__(self, inputs: typing.List[Type], outputs: typing.List[Type]):
        self.inputs = inputs
        self.outputs = outputs


