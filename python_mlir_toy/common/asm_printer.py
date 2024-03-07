import sys
import typing

from python_mlir_toy.common import mlir_type, tools


class Indent:
    def __init__(self, level=0):
        self.level = 0

    def __enter__(self):
        self.level += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.level -= 1

    def __str__(self):
        return '  ' * self.level

    def dump(self):
        print(self, end='')


class TemplatedSymbol:
    def __init__(self, identifier_list: typing.List[str], template_list: typing.List['TemplatedSymbol']):
        self.identifier_list = identifier_list
        self.template_list = template_list

    def __str__(self):
        ret = f'{".".join(self.identifier_list)}'
        if len(self.template_list) != 0:
            ret += f'<{", ".join(str(i) for i in self.template_list)}>'
        return ret


class TemplatedSymbolParser:
    def __init__(self, templated_symbol_str: str):
        self.templated_symbol_str = templated_symbol_str
        self.position = 0
        self.last_char = templated_symbol_str[0] if templated_symbol_str else None
        self.last_token = None
        self.move_next_token()

    def get_char(self) -> str:
        return self.last_char

    def move_next_char(self) -> None:
        self.position += 1
        if self.position < len(self.templated_symbol_str):
            self.last_char = self.templated_symbol_str[self.position]
        else:
            self.last_char = None
            self.position = len(self.templated_symbol_str)

    def skip_space(self) -> None:
        while self.get_char() is not None and self.get_char().isspace():
            self.move_next_char()

    def get_token(self):
        return self.last_token

    def move_next_token(self):
        self.skip_space()
        if self.get_char() is None:
            self.last_token = None
        elif self.get_char() in ',.<>':
            self.last_token = self.get_char()
            self.move_next_char()
        elif self.get_char().isidentifier():
            identifier = ''
            while self.get_char() is not None and (identifier + self.get_char()).isidentifier():
                identifier += self.get_char()
                self.move_next_char()
            self.last_token = identifier
        else:
            raise ValueError(f'Unknown character: "{self.get_char()}" in {self.templated_symbol_str}[{self.position}]')

    def parse(self) -> TemplatedSymbol:
        identifier_list = []
        template_list = []

        while self.get_token() is not None and self.get_token() not in '<>,':
            if self.get_token() == '.':
                pass
            elif self.get_token() == '<':
                break
            else:
                identifier_list.append(self.get_token())
            self.move_next_token()

        if self.get_token() == '<':
            self.move_next_token()
            while self.get_token() != '>':
                template_list.append(self.parse())
                assert self.get_token() in ',>'
                if self.get_token() == ',':
                    self.move_next_token()

            assert self.get_token() == '>'
            self.move_next_token()

        return TemplatedSymbol(identifier_list, template_list)

    def __str__(self):
        return str(self.parse())


class AsmPrinter:
    def __init__(self, output_file_like: typing.TextIO = None, ident: Indent = None,
                 flags: typing.Dict[str, typing.Any] = None):
        self.file = sys.stdout if output_file_like is None else output_file_like
        self.ident = Indent() if ident is None else ident
        self.flags: typing.Dict[str, typing.Any] = {} if flags is None else flags

    def get_flag(self, name: str):
        return self.flags.get(name)

    def print(self, *args, sep=' ', end=''):
        print(*args, sep, end, file=self.file)

    def print_float(self, value: float):
        self.print(str(value))

    def print_dialect_symbol(self, prefix: str, dialect_name: str, symbol_name: str):
        self.print(f'{prefix}{dialect_name}.{symbol_name}')

    def print_dialect_attr(self, dialect_name: str, attr_name: str):
        self.print_dialect_symbol('#', dialect_name, attr_name)

    def print_dialect_type(self, dialect_name: str, attr_name: str):
        self.print_dialect_symbol('!', dialect_name, attr_name)

    def print_alias(self, ty: mlir_type.Type) -> bool:
        return False

    def print_type(self, ty: mlir_type.Type):
        if ty is None:
            self.print('<<NULL TYPE>>')
        elif self.print_alias(ty):
            return
        elif isinstance(ty, mlir_type.OpaqueType):
            self.print_dialect_symbol('!', ty.dialect, ty.type_name)
        elif isinstance(ty, mlir_type.IndexType):
            self.print('index')
        elif isinstance(ty, mlir_type.IntegerType):
            self.print(f'{"s" if ty.signed else "u"}{ty.bits}i')
        elif isinstance(ty, mlir_type.Float32Type):
            self.print('f32')
        elif isinstance(ty, mlir_type.Float64Type):
            self.print('f64')
        elif isinstance(ty, mlir_type.FunctionType):
            self.print('(')
            for input_ty in tools.with_sep(ty.inputs, lambda: self.print(', ')):
                self.print_type(input_ty)
            self.print(') -> ')
            if len(ty.outputs) == 1:
                self.print_type(ty.outputs[0])
            else:
                self.print('(')
                for output_ty in tools.with_sep(ty.outputs, lambda: self.print(', ')):
                    self.print_type(output_ty)
                self.print(')')
        elif isinstance(ty, mlir_type.VectorType):
            self.print('vector<')
            for idx, dim in enumerate(tools.with_sep(ty.dims, lambda: self.print(', '))):
                if ty.is_scalable_dims is not None and ty.is_scalable_dims[idx]:
                    self.print(f'[{dim}]')
                else:
                    self.print(dim)
            self.print(f'x{self.print_type(ty.element_type)}>')
        elif isinstance(ty, mlir_type.RankedTensorType):
            self.print('tensor<')
            for dim in tools.with_sep(ty.shape, lambda: self.print('x')):
                self.print('?' if dim < 0 else str(dim))
            self.print(f'x{self.print_type(ty.element_type)}>')
        elif isinstance(ty, mlir_type.TensorType):
            self.print(f'tensor<*x{self.print_type(ty.element_type)}>')
        elif isinstance(ty, mlir_type.ComplexType):
            self.print(f'complex<{self.print_type(ty.element_type)}>')
        elif isinstance(ty, mlir_type.TupleType):
            self.print('tuple<')
            for input_ty in tools.with_sep(ty.types, lambda: self.print(', ')):
                self.print_type(input_ty)
            self.print('>')
        elif isinstance(ty, mlir_type.NoneType):
            self.print('none')
        else:
            raise NotImplementedError(f'Unsupported type: {ty}')

    def print_optional_attrs(self, attrs):
        raise NotImplementedError('TODO')

    def print_named_attr(self, name, value=None):
        if value is not None:
            self.print(name, '=', value)
        else:
            self.print(name)

    def print_escaped_str(self, string: str):
        escaped = string.replace('\\', '\\\\').replace('"', '\\"')
        self.print(f'"{escaped}"')

    def print_hex_str(self, string: str):
        self.print(f'hex"{string}"')

    def print_newline(self):
        self.print(end='\n')


class AsmParser:
    pass
