import typing

from python_mlir_toy.ch1 import ast, lexer
from python_mlir_toy.ch2 import ops
from python_mlir_toy.common import location, mlir_type, td, symbol_table, mlir_op


class MlirGenImpl:
    def __init__(self):
        self.func_dict: typing.Dict[str, ops.FuncOp] = {}
        self.symbol_table = symbol_table.SymbolTable()
        self.insert_point_list = []

    def insert_op(self, op: mlir_op.Op):
        self.insert_point_list[-1].append(op)
        return op

    @staticmethod
    def location(loc: lexer.Location):
        return location.FileLineColLocation(loc.filename, loc.line, loc.column)

    @staticmethod
    def op_to_value(op: mlir_op.Op) -> td.Value:
        assert isinstance(op.results, list)
        assert len(op.results) == 1
        return op.results[0]

    def mlir_gen(self, x):
        if isinstance(x, ast.ModuleAST):
            return self.mlir_gen_module(x)
        elif isinstance(x, ast.FunctionAST):
            return self.mlir_gen_func(x)
        elif isinstance(x, ast.CallExprAST):
            return self.mlir_gen_call(x)
        elif isinstance(x, ast.LiteralExprAST):
            return self.mlir_gen_literal(x)
        elif isinstance(x, ast.BinaryExprAST):
            return self.mlir_gen_binary(x)
        elif isinstance(x, ast.VariableExprAST):
            return self.mlir_gen_variable(x)
        elif isinstance(x, ast.VarDeclExprAST):
            return self.mlir_gen_var_decl(x)
        elif isinstance(x, ast.PrintExprAST):
            return self.mlir_gen_print(x)
        elif isinstance(x, ast.ReturnExprAST):
            return self.mlir_gen_return(x)
        else:
            raise NotImplementedError

    def mlir_gen_module(self, module: ast.ModuleAST):
        loc = location.UnknownLocation()

        for f in module.functions:
            self.mlir_gen_func(f)

        root_module = mlir_op.ModuleOp(loc, 'unknown', self.func_dict)

        # todo: verify module

        return root_module

    def mlir_gen_block(self, block: ast.ExprASTList, op_list):
        self.insert_point_list.append(op_list)
        for expr in block:
            self.mlir_gen(expr)
        self.insert_point_list.pop(-1)
        return op_list

    def mlir_gen_func(self, func: ast.FunctionAST):
        loc = self.location(func.location)

        func_input_types = [mlir_type.F64TensorType() for _ in func.proto.args]

        block = mlir_op.Block(func_input_types)

        with self.symbol_table:
            for name_ast, argument_value in zip(func.proto.args, block.arguments):
                self.symbol_table.insert(name_ast.name, argument_value)
            self.mlir_gen_block(func.body, block.op_list)

        # fixme: assume no branch
        if isinstance(block.op_list[-1], ops.ReturnOp):
            result_op = block.op_list[-1]
            func_output_types = [i.ty for i in result_op.operands]
        else:
            func_output_types = []

        func_type = mlir_type.FunctionType(func_input_types, func_output_types)
        ret = ops.FuncOp(loc, func.proto.name, func_type, block)
        self.func_dict[func.proto.name] = ret
        return ret

    def mlir_gen_call(self, call: ast.CallExprAST):
        loc = self.location(call.location)
        if call.callee == 'transpose':
            permutation = [1, 0]
            ret = ops.TransposeOp(loc, permutation, self.mlir_gen(call.args[0]))
        else:
            callee = self.func_dict[call.callee]
            inputs = [self.mlir_gen(arg) for arg in call.args]
            ret = ops.GenericCallOp(loc, callee, *inputs)
        self.insert_op(ret)
        return ret

    def mlir_gen_literal(self, literal: ast.LiteralExprAST):
        loc = self.location(literal.location)
        values = []

        def flatten(x):
            if isinstance(x, ast.LiteralExprAST):
                for item in x.values:
                    flatten(item)
            elif isinstance(x, ast.NumberExprAST):
                values.append(x.value)
            else:
                raise NotImplementedError('unknown literal type: %s' % type(x))

        flatten(literal)
        ret = ops.ConstantOp(loc, literal.dims, values)
        self.insert_op(ret)
        return ret

    def mlir_gen_number(self, literal: ast.NumberExprAST):
        loc = self.location(literal.location)
        ret = ops.ConstantOp(loc, [], [literal.value])
        self.insert_op(ret)
        return ret

    def mlir_gen_binary(self, binop: ast.BinaryExprAST):
        loc = self.location(binop.location)
        lhs = self.op_to_value(self.mlir_gen(binop.lhs))
        rhs = self.op_to_value(self.mlir_gen(binop.rhs))
        # assert isinstance(lhs, td.Value) and isinstance(lhs.ty, int)
        # assert isinstance(rhs, td.Value) and isinstance(rhs.ty, int)

        if binop.op == '+':
            ret = ops.AddOp(loc, lhs, rhs)
        elif binop.op == '*':
            ret = ops.MulOp(loc, lhs, rhs)
        else:
            raise NotImplementedError

        self.insert_op(ret)
        return ret

    def mlir_gen_variable(self, var: ast.VariableExprAST):
        ret = self.symbol_table.lookup(var.name)
        assert ret is not None
        return ret

    def mlir_gen_var_decl(self, decl: ast.VarDeclExprAST):
        loc = self.location(decl.location)
        var_decl_op = self.mlir_gen(decl.init_value)
        self.symbol_table.insert(decl.name, self.op_to_value(var_decl_op))
        if decl.var_type is None or decl.var_type.shape is None or len(decl.var_type.shape) == 0:
            return var_decl_op
        else:
            shape = decl.var_type.shape
            reshape_op = ops.ReshapeOp(loc, shape, var_decl_op)
            self.insert_op(reshape_op)
            return reshape_op

    def mlir_gen_print(self, p: ast.PrintExprAST):
        loc = self.location(p.location)
        operand = self.mlir_gen(p.content)
        ret = ops.PrintOp(loc, operand)
        self.insert_op(ret)
        return ret

    def mlir_gen_return(self, ret: ast.ReturnExprAST):
        loc = self.location(ret.location)
        if ret.expr is None:
            return ops.ReturnOp(loc)
        operand = self.op_to_value(self.mlir_gen(ret.expr))
        ret = ops.ReturnOp(loc, operand)
        self.insert_op(ret)
        return ret
