import typing

from python_mlir_toy.common import serializable, scoped_text_parser, scoped_text_printer, tools, td, mlir_type, \
    location, mlir_literal


def parse_namespaced_symbol(src: serializable.TextParser) -> str:
    symbol = src.last_token()
    src.drop_token(check_kind=serializable.TokenKind.Identifier, skip_space=False)
    while src.last_token() == '.':
        src.drop_token(skip_space=False)
        symbol += '.' + src.last_token()
        src.drop_token(check_kind=serializable.TokenKind.Identifier)
    src.drop_space()
    return symbol


class Format:
    def print(self, obj, dst: serializable.TextPrinter) -> None:
        # dst.print(self.getter(obj))
        raise NotImplementedError('print is not implemented')

    def parse(self, attr_dict, src: serializable.TextParser) -> None:
        # ret = self.setter(obj, src.last_token())
        # src.drop_token()
        # return ret
        raise NotImplementedError('parse is not implemented')


class ConstantStrFormat(Format):
    def __init__(self, text: str):
        self.text = text

    def print(self, obj, dst: serializable.TextPrinter) -> None:
        dst.print(self.text)

    def parse(self, attr_dict, src: serializable.TextParser) -> None:
        src.drop_token(check_token=self.text.strip())


class BoundedStrFormat(Format):
    def __init__(self, attr_name: str, prefix: str = None):
        self.attr_name = attr_name
        self.prefix = prefix

    def print(self, op, dst: serializable.TextPrinter) -> None:
        value = getattr(op, self.attr_name)
        assert isinstance(value, str)
        if self.prefix is not None:
            dst.print(self.prefix, end='')
        dst.print(value)

    def parse(self, attr_dict, src: serializable.TextParser) -> None:
        src.drop_token(check_token=self.prefix, skip_space=False)
        value = src.last_token()
        src.drop_token(check_kind=serializable.TokenKind.String)
        attr_dict[self.attr_name] = value


class BoundedNumberAttrFormat(Format):
    def __init__(self, attr_name: str):
        self.attr_name = attr_name

    def print(self, op, dst: serializable.TextPrinter) -> None:
        value = getattr(op, self.attr_name)
        assert isinstance(value, (int, float))
        dst.print(value)

    def parse(self, attr_dict, src: serializable.TextParser) -> None:
        last_token = src.last_token()
        src.drop_token(check_kind=serializable.TokenKind.Number)
        attr_dict[self.attr_name] = last_token


class BoundedLiteralAttrFormat(Format):
    def __init__(self, attr_name: str):
        self.attr_name = attr_name

    def print(self, op, dst: serializable.TextPrinter) -> None:
        value = getattr(op, self.attr_name)
        assert isinstance(value, mlir_literal.Literal)
        value.print(dst)

    def parse(self, attr_dict, src: serializable.TextParser) -> None:
        value = mlir_literal.parse_literal(src)
        attr_dict[self.attr_name] = value


class BoundedInputFormat(Format):
    def __init__(self, attr_name: str):
        self.attr_name = attr_name

    def print(self, op, dst: scoped_text_printer.ScopedTextPrinter) -> None:
        input_val = getattr(op, self.attr_name)
        assert isinstance(input_val, td.Value)
        input_name = dst.lookup_value_name(input_val)
        assert isinstance(input_name, str)
        assert input_name.startswith('%')
        dst.print(input_name, end='')

    def parse(self, attr_dict: typing.Dict, src: scoped_text_parser.ScopedTextParser) -> None:
        src.drop_token('%', skip_space=False)
        variable_name = str(src.last_token())
        src.drop_token()
        input_name = '%' + variable_name
        input_val = src.lookup_var(input_name)
        assert input_val is not None
        attr_dict[self.attr_name] = input_val


class BoundedOptionalInputFormat(Format):
    def __init__(self, attr_name: str):
        self.attr_name = attr_name

    def print(self, op, dst: scoped_text_printer.ScopedTextPrinter) -> None:
        input_val = getattr(op, self.attr_name)
        if input_val is not None:
            assert isinstance(input_val, td.Value)
            input_name = dst.lookup_value_name(input_val)
            assert isinstance(input_name, str)
            assert input_name.startswith('%')
            dst.print(input_name, end='')

    def parse(self, attr_dict, src: scoped_text_parser.ScopedTextParser) -> None:
        if src.last_token() == '%':
            src.drop_token('%', skip_space=False)
            operand_name = str(src.last_token())
            src.drop_token()
            input_name = '%' + operand_name
            input_val = src.lookup_var(input_name)
            assert input_val is not None
            attr_dict[self.attr_name] = input_val


class InputsFormat(Format):
    def print(self, obj, dst: scoped_text_printer.ScopedTextPrinter) -> None:
        inputs = obj.get_inputs()
        for input_val in tools.with_sep(inputs, sep=lambda: dst.print(', ')):
            assert isinstance(input_val, td.Value)
            input_name = dst.lookup_value_name(input_val)
            assert input_name is not None
            assert input_name.startswith('%')
            dst.print(input_name, end='')

    @staticmethod
    def parse_input(src: scoped_text_parser.ScopedTextParser):
        src.drop_token('%', skip_space=False)
        variable_name = str(src.last_token())
        src.drop_token()
        input_name = '%' + variable_name
        input_val = src.lookup_var(input_name)
        assert input_val is not None
        return input_val

    def parse(self, attr_dict, src: scoped_text_parser.ScopedTextParser) -> None:
        inputs = [self.parse_input(src)]
        while src.last_token() == ',':
            src.drop_token()
            inputs.append(self.parse_input(src))
        attr_dict['inputs'] = inputs


class BoundedTypeFormat(Format):
    def __init__(self, attr_name: str, prefix: str | None = ':', end: str = ' '):
        self.attr_name = attr_name
        self.prefix = prefix
        self.end = end
        assert self.end is None or self.end.isspace()

    def print(self, op, dst: serializable.TextPrinter) -> None:
        value = getattr(op, self.attr_name)
        if value is not None:
            if self.prefix is not None:
                dst.print(self.prefix)
            assert isinstance(value, td.Value)
            value_type = value.ty
            assert isinstance(value_type, mlir_type.Type)
            value_type.print(dst)
            if self.end is not None:
                dst.print(self.end)

    def parse(self, attr_dict: typing.Dict, src: serializable.TextParser) -> None:
        if self.prefix is not None:
            if src.last_token() == self.prefix.strip():
                src.drop_token()
            else:
                return

        ty = mlir_type.parse_type(src)
        attr_dict[self.attr_name + '_type'] = ty


class OutputsTypeFormat(Format):
    def __init__(self, prefix: str | None = ':', sep: str = None, parentheses_required: bool = False):
        self.prefix = prefix
        self.sep = sep or ','
        self.parentheses_required = parentheses_required

    def print(self, op, dst: serializable.TextPrinter) -> None:
        if self.prefix is not None:
            dst.print(self.prefix)
        results_type = [item.ty for item in op.get_outputs()]
        mlir_type.print_type_list(dst, results_type, self.sep, self.parentheses_required)

    def parse(self, attr_dict: typing.Dict, src: serializable.TextParser) -> None:
        if self.prefix is not None:
            if src.last_token() == self.prefix.strip():
                src.drop_token()
            else:
                return

        output_types = mlir_type.parse_type_list(src, self.sep.strip())
        attr_dict['output_types'] = output_types


class BoundedFunctionTypeFormat(Format):
    def __init__(self, attr_name: str, prefix: str = None):
        self.prefix = prefix or ':'
        self.attr_name = attr_name

    def print(self, op, dst: serializable.TextPrinter) -> None:
        if self.prefix is not None:
            dst.print(self.prefix)
        func = getattr(op, self.attr_name)
        assert isinstance(func.function_type, mlir_type.FunctionType)
        func.function_type.print(dst)

    def parse(self, attr_dict: typing.Dict, src: serializable.TextParser) -> None:
        if src.last_token() == self.prefix.strip():
            src.drop_token()
            function_type = mlir_type.parse_function_type(src)
            attr_dict[self.attr_name + '_type'] = function_type


class LocationFormat(Format):
    def print(self, op, dst: serializable.TextPrinter) -> None:
        if op.loc is not None:
            op.loc.print(dst)

    def parse(self, attr_dict: typing.Dict, src: serializable.TextParser) -> None:
        attr_dict['loc'] = location.parse_location(src)


class FunctionDeclarationFormat(Format):
    def __init__(self, op_builder):
        self.op_builder = op_builder

    def print(self, op, dst: scoped_text_printer.ScopedTextPrinter) -> None:
        assert isinstance(op.function_name, str)
        assert op.function_name.startswith('@')
        assert isinstance(op.function_type, mlir_type.FunctionType)
        dst.insert_value_name(op, op.function_name)

        dst.print(op.function_name)
        dst.print('(', end='')
        for arg_name, arg_type, arg_loc in tools.with_sep(
                zip(op.argument_names, op.function_type.inputs, op.argument_locs), lambda: dst.print(',')
        ):
            assert isinstance(arg_name, str)
            assert isinstance(arg_type, mlir_type.Type)
            assert isinstance(arg_loc, location.Location)
            dst.print(arg_name, ':', sep='')
            arg_type.print(dst)
            dst.print()
            if arg_loc is not None:
                arg_loc.print(dst)
        dst.print(')')
        if len(op.function_type.outputs) > 0:
            dst.print('->')
            mlir_type.print_type_list(dst, op.function_type.outputs, sep=',')

        assert isinstance(op.body, typing.List)
        dst.print('{')
        dst.print_newline()
        with dst:
            for arg_name, arg_value in zip(op.argument_names, op.argument_values):
                dst.insert_value_name(arg_value, arg_name)

            for item in op.body:
                dst.print_ident()
                output_names = [dst.insert_value_and_generate_name(item) for item in item.get_outputs()]
                if len(output_names) > 0:
                    for output_name in tools.with_sep(output_names, lambda: dst.print(',')):
                        dst.print(output_name, end='')

                    dst.print(' = ', end='')
                dst.print(item.op_name)
                item.print(dst)
                dst.print_newline()

        dst.print_ident()
        dst.print('}')

    def parse(self, attr_dict: typing.Dict, src: scoped_text_parser.ScopedTextParser) -> None:
        src.drop_token('@', skip_space=False)
        function_name = '@' + src.last_token()
        src.drop_token()
        arg_names = []
        arg_types = []
        arg_locs = []
        src.drop_token('(')
        while src.last_token() != ')':
            if src.last_token() == ',':
                src.drop_token()
            src.drop_token('%')
            arg_names.append('%' + src.last_token())
            src.drop_token()
            src.drop_token(':')
            arg_types.append(mlir_type.parse_type(src))
            arg_locs.append(location.parse_location(src))

        src.drop_token(')')

        if src.last_token() == '-':
            src.drop_token('-')
            src.drop_token('>')
            output_types = mlir_type.parse_type_list(src)
        else:
            output_types = []

        function_type = mlir_type.FunctionType(arg_types, output_types)

        src.drop_token('{')
        arg_vals = [td.Value(ty) for ty in arg_types]
        for arg_name, arg_val in zip(arg_names, arg_vals):
            src.define_var(arg_name, arg_val)

        body = []
        while src.last_token() != '}':
            output_names = []
            if src.last_token() == '%':
                while src.last_token() != '=':
                    if src.last_token() == ',':
                        src.drop_token()
                    src.drop_token('%')
                    output_names.append('%' + str(src.last_token()))
                    src.drop_token()
                src.drop_token('=')

            op_name = parse_namespaced_symbol(src)
            op_cls = self.op_builder(op_name)
            op = op_cls.parse(src)
            body.append(op)

            op_outputs = op.get_outputs()
            assert len(op_outputs) == len(output_names)
            for output_name, output_value in zip(output_names, op_outputs):
                src.define_var(output_name, output_value)

        src.drop_token('}')

        attr_dict['function_type'] = function_type
        attr_dict['function_name'] = function_name
        attr_dict['argument_names'] = arg_names
        attr_dict['argument_values'] = arg_vals
        attr_dict['argument_locs'] = arg_locs
        attr_dict['body'] = body


class CalleeFormat(Format):
    def print(self, obj, dst: scoped_text_printer.ScopedTextPrinter) -> None:
        callee = obj.callee
        callee_name = dst.lookup_value_name(callee)
        assert callee_name.startswith('@')
        dst.print(callee_name)

    def parse(self, attr_dict: typing.Dict, src: scoped_text_parser.ScopedTextParser) -> None:
        src.drop_token('@', skip_space=False)
        callee_name = '@' + src.last_token()
        src.drop_token()
        callee = src.lookup_var(callee_name)
        attr_dict['callee'] = callee


class ModuleDeclarationFormat(Format):
    def __init__(self, op_builder):
        self.op_builder = op_builder

    def print(self, op, dst: serializable.TextPrinter) -> None:
        dst.print('module {')
        dst.print_newline()
        with dst:
            for item in op.body:
                dst.print(item.op_name)
                item.print(dst)
                dst.print_newline()
        dst.print('}')

    def parse(self, attr_dict: typing.Dict, src: scoped_text_parser.ScopedTextParser) -> None:
        src.drop_token('module')
        src.drop_token('{')
        body = []
        while src.last_token() != '}':
            op_name = parse_namespaced_symbol(src)
            op_cls = self.op_builder(op_name)
            op = op_cls.parse(src)
            body.append(op)
            src.define_var(op.function_name, op)
        src.drop_token('}')
        attr_dict['body'] = body
