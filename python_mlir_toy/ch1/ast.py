import enum
from typing import Optional

from python_mlir_toy.ch1.lexer import Location


class VarType:
    def __init__(self, shape: [int]):
        self.shape = shape


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


ExprASTList: [ExprAST] = list


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
        self.arg = arg


class PrototypeAST:
    def __init__(self, location: Location, name: str, args: [VariableExprAST]):
        self.location = location
        self.name = name
        self.args = args


class FunctionAST:
    def __init__(self, location: Location, proto: PrototypeAST, body: ExprAST):
        self.location = location
        self.proto = proto
        self.body = body


class ModuleAST:
    def __init__(self, location: Location, functions: [FunctionAST]):
        self.location = location
        self.functions = functions


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


class ASTDumper:
    def __init__(self):
        self.indent = Indent()

    def dump(self, module: ModuleAST):
        with self.indent:
            self.indent.dump()
            print('Module:')
            for f in module.functions:
                # f.dump()
                print('<function>')


def dump(module: ModuleAST):
    dumper = ASTDumper()
    dumper.dump(module)
