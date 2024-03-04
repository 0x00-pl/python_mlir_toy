from python_mlir_toy.ch1 import ast, lexer
from python_mlir_toy.ch2 import ops
from python_mlir_toy.common import mlir_op, location, mlir_type


class MlirGenImpl:
    def __init__(self):
        self.root_module = None

    def location(self, loc: lexer.Location):
        return location.FileLineColLocation(loc.filename, loc.line, loc.column)

    def mlir_gen(self, x):
        if isinstance(x, ast.ModuleAST):
            return self.mlir_gen_module(x)
        else:
            raise NotImplementedError

    def mlir_gen_module(self, module):
        loc = location.UnknownLocation()
        self.root_module = mlir_op.ModuleOp(loc)

        for f in module.functions:
            self.mlir_gen(f)

        # todo: verify module

        return self.root_module

    def mlir_gen_func(self, func: ast.FunctionAST):
        loc = self.location(func.location)

        func_type = mlir_type.FunctionType(
            [mlir_type.UnrankedTensorType(mlir_type.Float64Type()) for _ in func.proto.args],
            []
        )

        body = []
        func = ops.FuncOp(loc, func.proto.name, func_type, body)



        func.regions[0] = None




