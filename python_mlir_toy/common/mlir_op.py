import sys
import typing

from python_mlir_toy.common import serializable, td, location, scoped_text_printer, scoped_text_parser, bounded_format, \
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

    def get_inputs(self) -> typing.List[td.Value]:
        raise NotImplementedError('get_inputs is not implemented')

    def get_outputs(self) -> typing.List[td.Value]:
        raise NotImplementedError('get_outputs is not implemented')

    @classmethod
    def get_format_list(cls) -> typing.List[bounded_format.Format]:
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
    def __init__(
            self, loc: location.Location, inputs: typing.List[td.Value], output_types: typing.List[mlir_type.Type]
    ):
        super().__init__(loc=loc)
        self.inputs = inputs
        self.outputs = [td.Value(ty) for ty in output_types]

    def get_inputs(self) -> typing.List[td.Value]:
        return self.inputs

    def get_outputs(self) -> typing.List[td.Value]:
        return self.outputs

    @classmethod
    def get_format_list(cls):
        return [bounded_format.InputsFormat(), bounded_format.OutputsTypeFormat(), bounded_format.LocationFormat()]


class BinaryOp(Op):
    def __init__(
            self, loc: location.Location, lhs: td.Value, rhs: td.Value, output_types: typing.List[mlir_type.Type] = None
    ):
        super().__init__(loc=loc)
        self.lhs = lhs
        self.rhs = rhs
        assert lhs.ty == rhs.ty
        self.output = td.Value(output_types[0] if output_types is not None else lhs.ty)
        assert lhs.ty <= self.output.ty

    def get_inputs(self) -> typing.List[td.Value]:
        return [self.lhs, self.rhs]

    def get_outputs(self) -> typing.List[td.Value]:
        return [self.output]

    @classmethod
    def get_format_list(cls):
        return [bounded_format.BoundedInputFormat('lhs'), bounded_format.ConstantStrFormat(','),
                bounded_format.BoundedInputFormat('rhs'), bounded_format.OutputsTypeFormat(),
                bounded_format.LocationFormat()]


class FuncOp(Op):
    op_name = 'func'

    def __init__(
            self, loc: location.Location, function_type: mlir_type.FunctionType, function_name: str,
            argument_names: typing.List[str], argument_values: typing.List[td.Value],
            argument_locs: typing.List[location.Location], body: typing.List[Op]
    ):
        super().__init__(loc=loc)
        assert len(function_type.inputs) == len(argument_names)
        assert len(argument_locs) == len(argument_names)
        self.function_type = function_type
        self.function_name = function_name
        self.argument_names = argument_names
        self.argument_locs = argument_locs
        self.argument_values = argument_values
        self.body = body

    @classmethod
    def get_format_list(cls):
        return [bounded_format.FunctionDeclarationFormat(Op.get_op_cls), bounded_format.LocationFormat()]


class GenericCallOp(Op):
    op_name = 'generic_call'

    def __init__(
            self, loc: location.Location, callee: FuncOp, inputs: typing.List[td.Value],
            callee_type: mlir_type.FunctionType = None
    ):
        super().__init__(loc=loc)
        self.callee = callee
        self.inputs = inputs
        self.outputs = [td.Value(ty) for ty in callee.function_type.outputs]
        assert len(inputs) == len(callee.function_type.inputs)
        assert callee_type is None or callee_type <= callee.function_type

    def get_inputs(self) -> typing.List[td.Value]:
        return self.inputs

    def get_outputs(self) -> typing.List[td.Value]:
        return self.outputs

    @classmethod
    def get_format_list(cls):
        return [bounded_format.CalleeFormat(), bounded_format.ConstantStrFormat('('), bounded_format.InputsFormat(),
                bounded_format.ConstantStrFormat(')'), bounded_format.CalleeFunctionTypeFormat(),
                bounded_format.LocationFormat()]


class ModuleOp(Op):
    op_name = 'module'

    def __init__(self, loc: location.Location, body: typing.List[Op]):
        super().__init__(loc=loc)
        self.body = body

    @classmethod
    def get_format_list(cls):
        return [bounded_format.ModuleDeclarationFormat(Op.get_op_cls), bounded_format.LocationFormat()]

    def dump(self):
        printer = scoped_text_printer.ScopedTextPrinter(file=sys.stdout)
        self.print(printer)
        printer.print_newline()


def parse_module(src: scoped_text_parser.ScopedTextParser):
    return ModuleOp.parse(src)
