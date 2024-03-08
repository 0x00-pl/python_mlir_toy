# class AsmPrinter:
#     def __init__(self, output_file_like: typing.TextIO = None, ident: serializable.Indent = None,
#                  flags: typing.Dict[str, typing.Any] = None):
#         self.file = sys.stdout if output_file_like is None else output_file_like
#         self.ident = serializable.Indent() if ident is None else ident
#         self.flags: typing.Dict[str, typing.Any] = {} if flags is None else flags
#
#     def get_flag(self, name: str):
#         return self.flags.get(name)
#
#     def print(self, *args, sep=' ', end=''):
#         print(*args, sep, end, file=self.file)
#
#     def print_float(self, value: float):
#         self.print(str(value))
#
#     def print_dialect_symbol(self, prefix: str, dialect_name: str, symbol_name: str):
#         self.print(f'{prefix}{dialect_name}.{symbol_name}')
#
#     def print_dialect_attr(self, dialect_name: str, attr_name: str):
#         self.print_dialect_symbol('#', dialect_name, attr_name)
#
#     def print_dialect_type(self, dialect_name: str, attr_name: str):
#         self.print_dialect_symbol('!', dialect_name, attr_name)
#
#     def print_alias(self, ty: mlir_type.Type) -> bool:
#         return False
#
#     def print_type(self, ty: mlir_type.Type):
#         if ty is None:
#             self.print('<<NULL TYPE>>')
#         elif self.print_alias(ty):
#             return
#         elif isinstance(ty, mlir_type.OpaqueType):
#             self.print_dialect_symbol('!', ty.dialect, ty.type_name)
#         elif isinstance(ty, mlir_type.IndexType):
#             self.print('index')
#         elif isinstance(ty, mlir_type.IntegerType):
#             self.print(f'{"s" if ty.signed else "u"}{ty.bits}i')
#         elif isinstance(ty, mlir_type.Float32Type):
#             self.print('f32')
#         elif isinstance(ty, mlir_type.Float64Type):
#             self.print('f64')
#         elif isinstance(ty, mlir_type.FunctionType):
#             self.print('(')
#             for input_ty in tools.with_sep(ty.inputs, lambda: self.print(', ')):
#                 self.print_type(input_ty)
#             self.print(') -> ')
#             if len(ty.outputs) == 1:
#                 self.print_type(ty.outputs[0])
#             else:
#                 self.print('(')
#                 for output_ty in tools.with_sep(ty.outputs, lambda: self.print(', ')):
#                     self.print_type(output_ty)
#                 self.print(')')
#         elif isinstance(ty, mlir_type.VectorType):
#             self.print('vector<')
#             for idx, dim in enumerate(tools.with_sep(ty.dims, lambda: self.print(', '))):
#                 if ty.is_scalable_dims is not None and ty.is_scalable_dims[idx]:
#                     self.print(f'[{dim}]')
#                 else:
#                     self.print(dim)
#             self.print(f'x{self.print_type(ty.element_type)}>')
#         elif isinstance(ty, mlir_type.RankedTensorType):
#             self.print('tensor<')
#             for dim in tools.with_sep(ty.shape, lambda: self.print('x')):
#                 self.print('?' if dim < 0 else str(dim))
#             self.print(f'x{self.print_type(ty.element_type)}>')
#         elif isinstance(ty, mlir_type.TensorType):
#             self.print(f'tensor<*x{self.print_type(ty.element_type)}>')
#         elif isinstance(ty, mlir_type.ComplexType):
#             self.print(f'complex<{self.print_type(ty.element_type)}>')
#         elif isinstance(ty, mlir_type.TupleType):
#             self.print('tuple<')
#             for input_ty in tools.with_sep(ty.types, lambda: self.print(', ')):
#                 self.print_type(input_ty)
#             self.print('>')
#         elif isinstance(ty, mlir_type.NoneType):
#             self.print('none')
#         else:
#             raise NotImplementedError(f'Unsupported type: {ty}')
#
#     def print_optional_attrs(self, attrs):
#         raise NotImplementedError('TODO')
#
#     def print_named_attr(self, name, value=None):
#         if value is not None:
#             self.print(name, '=', value)
#         else:
#             self.print(name)
#
#     def print_escaped_str(self, string: str):
#         escaped = string.replace('\\', '\\\\').replace('"', '\\"')
#         self.print(f'"{escaped}"')
#
#     def print_hex_str(self, string: str):
#         self.print(f'hex"{string}"')
#
#     def print_newline(self):
#         self.print(end='\n')


# class AsmParser:
#     pass
