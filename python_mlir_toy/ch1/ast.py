import enum
from typing import Optional, List, Union

from python_mlir_toy.ch1.lexer import Location
from python_mlir_toy.common.asm_printer import Indent


class VarType:
    def __init__(self, shape: List[int]):
        self.shape: List[int] = shape

    def __str__(self):
        return f'<{", ".join(str(i) for i in self.shape)}>'


class ExprAST:
    class ExprASTKind(enum.Enum):
        Expr_VarDecl = 0
        Expr_Return = 1
        Expr_Num = 2
        Expr_Literal = 3
        Expr_Var = 4
        Expr_BinOp = 5
        Expr_Call = 6
        Expr_Print = 7

    def __init__(self, kind: ExprASTKind, location: Location):
        self.kind = kind
        self.location = location


ExprASTList = List[ExprAST]


class NumberExprAST(ExprAST):
    def __init__(self, location: Location, value: int):
        super().__init__(ExprAST.ExprASTKind.Expr_Num, location)
        self.value = value


class LiteralExprAST(ExprAST):
    def __init__(self, location: Location, values: ExprASTList, dims: [int]):
        super().__init__(ExprAST.ExprASTKind.Expr_Literal, location)
        self.values = values
        self.dims = dims


class VariableExprAST(ExprAST):
    def __init__(self, location: Location, name: str):
        super().__init__(ExprAST.ExprASTKind.Expr_Var, location)
        self.name = name


class VarDeclExprAST(ExprAST):
    def __init__(self, location: Location, name: str, var_type: VarType, init_value: ExprAST):
        super().__init__(ExprAST.ExprASTKind.Expr_VarDecl, location)
        self.name = name
        self.var_type = var_type
        self.init_value = init_value


class ReturnExprAST(ExprAST):
    def __init__(self, location: Location, expr: Optional[ExprAST] = None):
        super().__init__(ExprAST.ExprASTKind.Expr_Return, location)
        self.expr = expr


class BinaryExprAST(ExprAST):
    def __init__(self, location: Location, op: str, lhs: ExprAST, rhs: ExprAST):
        super().__init__(ExprAST.ExprASTKind.Expr_BinOp, location)
        self.op = op
        self.lhs = lhs
        self.rhs = rhs


class CallExprAST(ExprAST):
    def __init__(self, location: Location, callee: str, args: ExprASTList):
        super().__init__(ExprAST.ExprASTKind.Expr_Call, location)
        self.callee = callee
        self.args = args


class PrintExprAST(ExprAST):
    def __init__(self, location: Location, arg: ExprAST):
        super().__init__(ExprAST.ExprASTKind.Expr_Print, location)
        self.content = arg


class PrototypeAST:
    def __init__(self, location: Location, name: str, args: List[VariableExprAST]):
        self.location = location
        self.name = name
        self.args = args


class FunctionAST:
    def __init__(self, location: Location, proto: PrototypeAST, body: ExprASTList):
        self.location = location
        self.proto = proto
        self.body = body


class ModuleAST:
    def __init__(self, location: Location, functions: [FunctionAST]):
        self.location = location
        self.functions = functions


class ASTDumper:
    def __init__(self):
        self.indent = Indent()

    def dump(self, expr: ExprAST):
        if isinstance(expr, BinaryExprAST):
            self.dump_binary(expr)
        elif isinstance(expr, CallExprAST):
            self.dump_call(expr)
        elif isinstance(expr, LiteralExprAST):
            self.dump_literal(expr)
        elif isinstance(expr, NumberExprAST):
            self.dump_number(expr)
        elif isinstance(expr, PrintExprAST):
            self.dump_print(expr)
        elif isinstance(expr, ReturnExprAST):
            self.dump_return(expr)
        elif isinstance(expr, VarDeclExprAST):
            self.dump_var_decl(expr)
        elif isinstance(expr, VariableExprAST):
            self.dump_variable(expr)
        else:
            print(self.indent, 'unknown expression, kind:', expr.kind)

    def dump_var_type(self, var_type: VarType):
        print(var_type, end='')

    def dump_var_decl(self, expr: VarDeclExprAST):
        with self.indent:
            print(self.indent, 'VarDecl:', expr.name, end='')
            self.dump_var_type(expr.var_type)
            print('', expr.location)
            self.dump(expr.init_value)

    def dump_expr_list(self, expr_list: ExprASTList):
        with self.indent:
            print(self.indent, 'Block {')
            for expr in expr_list:
                self.dump(expr)
            print(self.indent, '} //Block')

    def dump_number(self, expr: NumberExprAST):
        with self.indent:
            print(expr.value, expr.location, end='')

    @staticmethod
    def print_literal_helper(literal: Union[NumberExprAST, LiteralExprAST]):
        if isinstance(literal, NumberExprAST):
            print(literal.value, end='')
        else:
            print(f'<{", ".join(str(i) for i in literal.dims)}>[', end='')
            first = True
            for value in literal.values:
                if first:
                    first = False
                else:
                    print(', ', end='')
                ASTDumper.print_literal_helper(value)

            print(']', end='')

    def dump_literal(self, expr: LiteralExprAST):
        with self.indent:
            print(self.indent, 'Literal: ', end='')
            self.print_literal_helper(expr)
            print('', expr.location)

    def dump_variable(self, expr: VariableExprAST):
        with self.indent:
            print(self.indent, 'Variable:', expr.name, expr.location)

    def dump_return(self, expr: ReturnExprAST):
        with self.indent:
            print(self.indent, 'Return')
            if expr.expr is not None:
                self.dump(expr.expr)
            else:
                with self.indent:
                    print(self.indent, '(void)')

    def dump_binary(self, expr: BinaryExprAST):
        with self.indent:
            print(self.indent, 'Binary:', expr.op, expr.location)
            self.dump(expr.lhs)
            self.dump(expr.rhs)

    def dump_call(self, expr: CallExprAST):
        with self.indent:
            print(self.indent, 'Call:', expr.callee, '[', expr.location)
            for arg in expr.args:
                self.dump(arg)
            print(self.indent, ']')

    def dump_print(self, expr: PrintExprAST):
        with self.indent:
            print(self.indent, 'Print [', expr.location)
            self.dump(expr.content)
            print(self.indent, ']')

    def dump_prototype(self, proto: PrototypeAST):
        with self.indent:
            print(self.indent, f'Prototype: "{proto.name}"', proto.location)
            print(self.indent, f'Params: [{", ".join(i.name for i in proto.args)}]')

    def dump_function(self, func: FunctionAST):
        with self.indent:
            print(self.indent, 'Function:')
            self.dump_prototype(func.proto)
            self.dump_expr_list(func.body)

    def dump_module(self, module: ModuleAST):
        print(self.indent, 'Module:')
        for f in module.functions:
            self.dump_function(f)


def dump(module: ModuleAST):
    dumper = ASTDumper()
    dumper.dump_module(module)
