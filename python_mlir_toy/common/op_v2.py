import typing

from python_mlir_toy.common import serializable, td, location, scoped_text_printer, scoped_text_parser, bonded_format, \
    mlir_type


class Op(serializable.TextSerializable):
    op_name: str = None
    op_type_dict: typing.Dict[str, typing.Type['Op']] = {}

    @staticmethod
    def register_op_cls(name: str, cls: typing.Type['Op']):
        assert name not in Op.op_type_dict
        Op.op_type_dict[name] = cls

    @staticmethod
    def get_op_cls(op_name: str) -> typing.Type['Op']:
        assert op_name in Op.op_type_dict
        return Op.op_type_dict[op_name]

    def __init_subclass__(cls):
        if cls.op_name is not None:
            Op.register_op_cls(cls.op_name, cls)

    def __init__(self, loc: location.Location):
        self.loc = loc

    def get_inputs(self):
        raise NotImplementedError('get_inputs is not implemented')

    def get_outputs(self):
        raise NotImplementedError('get_outputs is not implemented')

    @classmethod
    def get_format_list(cls) -> typing.List[bonded_format.BondedFormat]:
        raise NotImplementedError('get_format_list is not implemented')

    def print(self, dst: scoped_text_printer.ScopedTextPrinter):
        format_list = self.get_format_list()
        for format_item in format_list:
            format_item.print(self, dst)

    @classmethod
    def parse(cls, src: scoped_text_parser.ScopedTextParser):
        format_list = cls.get_format_list()
        attr_dict = {}
        for format_item in format_list:
            format_item.parse(attr_dict, src)
        return cls(**attr_dict)


class GeneralOp(Op):
    def __init__(self, loc: location.Location, operands: typing.List[td.Value],
                 output_types: typing.List[mlir_type.Type]):
        super().__init__(loc=loc)
        self.operands = operands
        self.outputs = [td.Value(ty) for ty in output_types]

    def get_inputs(self):
        return self.operands

    def get_outputs(self):
        return self.outputs

    @classmethod
    def get_format_list(cls):
        return [bonded_format.InputsFormat(), bonded_format.OutputsTypeFormat(), bonded_format.LocationTypeFormat()]


class BinaryOp(Op):
    def __init__(self, loc: location.Location, lhs: td.Value, rhs: td.Value):
        super().__init__(loc=loc)
        self.lhs = lhs
        self.rhs = rhs
        assert lhs.ty == rhs.ty
        self.result = td.Value(lhs.ty)

    def get_inputs(self):
        return [self.lhs, self.rhs]

    def get_outputs(self):
        return [self.result]

    @classmethod
    def get_format_list(cls):
        return [bonded_format.BoundedInputFormat('lhs'), bonded_format.ConstantStrFormat(','),
                bonded_format.BoundedInputFormat('rhs'), bonded_format.OutputsTypeFormat(),
                bonded_format.LocationTypeFormat()]


class AddOp(BinaryOp):
    op_name = 'toy.add'


class FuncOp(Op):
    op_name = 'func'

    def __init__(self, loc: location.Location, function_type: mlir_type.FunctionType, function_name: str,
                 argument_names: typing.List[str], argument_locs: typing.List[location.Location],
                 body: typing.List[Op]):
        super().__init__(loc=loc)
        assert len(function_type.inputs) == len(argument_names)
        assert len(argument_locs) == len(argument_names)
        self.function_type = function_type
        self.function_name = function_name
        self.argument_names = argument_names
        self.argument_locs = argument_locs
        self.body = body

    def get_inputs(self):
        return []

    def get_outputs(self):
        return []

    @classmethod
    def get_format_list(cls):
        return [
            bonded_format.FunctionDeclarationFormat(Op.get_op_cls),
            bonded_format.LocationTypeFormat()
        ]
