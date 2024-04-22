import typing

from python_mlir_toy.ch1 import ast, lexer
from python_mlir_toy.ch2 import ops
from python_mlir_toy.common import location, mlir_type, td, mlir_op, scoped, mlir_literal


class MlirGenImpl:
    def __init__(self):
        self.func_dict: typing.Dict[str, ops.ToyFuncOp] = {}
        self.symbol_table = scoped.SymbolTable[td.Value]()
        self.insert_point_list = []

    def insert_op(self, op: mlir_op.Op):
        self.insert_point_list[-1].append(op)
        return op

    @staticmethod
    def location(loc: lexer.Location):
        return location.FileLineColLocation(loc.filename, loc.line, loc.column)

    @staticmethod
    def op_to_value(op: mlir_op.Op) -> td.Value:
        output_values = op.get_outputs()
        assert isinstance(output_values, list)
        assert len(output_values) == 1
        return output_values[0]

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
        loc = self.location(module.location)

        for f in module.functions:
            self.mlir_gen_func(f)

        root_module = mlir_op.ModuleOp(loc, list(self.func_dict.values()))

        # todo: verify module

        return root_module

    def mlir_gen_block(self, block: ast.ExprASTList, op_list):
        self.insert_point_list.append(op_list)
        for expr in block:
            self.mlir_gen(expr)
        if not isinstance(op_list[-1], ops.ReturnOp):
            ret_op = ops.ReturnOp(op_list[-1].loc)
            self.insert_op(ret_op)
        self.insert_point_list.pop(-1)
        return op_list

    def mlir_gen_func(self, func: ast.FunctionAST):
        loc = self.location(func.location)

        func_input_types = [mlir_type.F64TensorType() for _ in func.proto.args]

        arguments = [td.Value(ty) for ty in func_input_types]
        op_list: typing.List[mlir_op.Op] = []
        arg_name_list: typing.List[str] = []
        arg_loc_list: typing.List[location.Location] = []

        with self.symbol_table:
            for name_ast, argument_value in zip(func.proto.args, arguments):
                assert isinstance(name_ast, ast.VariableExprAST)
                self.symbol_table.insert(name_ast.name, argument_value)
                arg_name_list.append('%' + name_ast.name)
                arg_loc_list.append(self.location(name_ast.location))
            self.mlir_gen_block(func.body, op_list)

        # fixme: assume no branch
        if isinstance(op_list[-1], ops.ReturnOp):
            result_op = op_list[-1]
            func_output_types = [i.ty for i in result_op.get_inputs()]
        else:
            func_output_types = []

        func_name = '@' + func.proto.name
        func_type = mlir_type.FunctionType(func_input_types, func_output_types)
        ret = ops.ToyFuncOp(loc, func_type, func_name, arg_name_list, arguments, arg_loc_list, op_list)
        self.func_dict[func.proto.name] = ret
        return ret

    def mlir_gen_call(self, call: ast.CallExprAST):
        loc = self.location(call.location)
        if call.callee == 'transpose':
            ret = ops.TransposeOp(loc, self.mlir_gen(call.args[0]))
        else:
            callee = self.func_dict[call.callee]
            inputs = [self.mlir_gen(arg) for arg in call.args]
            ret = ops.ToyGenericCallOp(loc, callee, inputs)
        self.insert_op(ret)
        return ret

    def mlir_gen_literal(self, literal: ast.LiteralExprAST):
        loc = self.location(literal.location)

        def to_array(x):
            if isinstance(x, ast.LiteralExprAST):
                return [to_array(item) for item in x.values]
            elif isinstance(x, ast.NumberExprAST):
                return x.value
            else:
                raise NotImplementedError('unknown literal type: %s' % type(x))

        values = to_array(literal)

        ret = ops.ConstantOp(loc, mlir_literal.DenseTensorLiteral(literal.dims, values))
        self.insert_op(ret)
        return ret

    def mlir_gen_binary(self, binop: ast.BinaryExprAST):
        loc = self.location(binop.location)
        lhs = self.op_to_value(self.mlir_gen(binop.lhs))
        rhs = self.op_to_value(self.mlir_gen(binop.rhs))

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
        ret_op = var_decl_op

        if isinstance(decl.init_value, ast.LiteralExprAST):
            var_type = var_decl_op.literal.get_type()
            if decl.var_type.shape is not None and len(decl.var_type.shape) != 0:
                if not (isinstance(var_type, mlir_type.RankedTensorType) and var_type.shape == decl.var_type.shape):
                    shape = decl.var_type.shape
                    literal_value = self.op_to_value(var_decl_op)
                    assert isinstance(literal_value.ty, mlir_type.RankedTensorType)
                    reshape_op = ops.ReshapeOp(
                        loc, literal_value, literal_value.ty,
                        mlir_type.RankedTensorType(literal_value.ty.element_type, shape)
                    )
                    self.insert_op(reshape_op)
                    ret_op = reshape_op

        self.symbol_table.insert(decl.name, self.op_to_value(ret_op))
        return ret_op

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
