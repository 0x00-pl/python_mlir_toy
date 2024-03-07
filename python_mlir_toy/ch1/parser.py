from typing import Optional

from python_mlir_toy.ch1.ast import FunctionAST, PrototypeAST, ExprASTList, VarDeclExprAST, VarType, VariableExprAST, \
    PrintExprAST, CallExprAST, NumberExprAST, LiteralExprAST, ExprAST, BinaryExprAST, ReturnExprAST, ModuleAST
from python_mlir_toy.ch1.lexer import Lexer, Token


class Parser:
    def __init__(self, lexer: Lexer):
        self.lexer = lexer

    def parse_error(self, expected, context):
        print(f'Parse error ({self.lexer.location}): '
              f'expected "{expected}" {context} but has Token: {self.lexer.get_cur_token()}')

    def parse_return(self):
        loc = self.lexer.location.copy()
        if self.lexer.get_cur_token() != Token.Return:
            return None
        self.lexer.consume(Token.Return)
        exp = self.parse_expression()
        if exp is None:
            return None
        return ReturnExprAST(loc, exp)

    def parse_identifier_expr(self):
        loc = self.lexer.location.copy()
        if self.lexer.get_cur_token() != Token.Identifier:
            return self.parse_error('identifier', 'in primary')
        name = self.lexer.identifier
        self.lexer.consume(Token.Identifier)

        if self.lexer.get_cur_token() != Token.ParenthesesOpen:
            return VariableExprAST(loc, name)

        if self.lexer.get_cur_token() != Token.ParenthesesOpen:
            return self.parse_error('(', 'in primary')
        self.lexer.consume(Token.ParenthesesOpen)

        args: ExprASTList = []
        while self.lexer.get_cur_token() != Token.ParenthesesClose:
            exp = self.parse_expression()
            if not exp:
                return None
            args.append(exp)
            if self.lexer.get_cur_token() not in (Token.Comma, Token.ParenthesesClose):
                return self.parse_error(', or )', 'in primary argument list')
            if self.lexer.get_cur_token() == Token.Comma:
                self.lexer.consume(Token.Comma)

        if self.lexer.get_cur_token() != Token.ParenthesesClose:
            return self.parse_error(')', 'in primary argument list')
        self.lexer.consume(Token.ParenthesesClose)

        if name == 'print':
            if len(args) != 1:
                return self.parse_error('<single argument>', 'as print(<arg>)')
            return PrintExprAST(loc, args[0])

        return CallExprAST(loc, name, args)

    def parse_number_expr(self):
        loc = self.lexer.location.copy()
        if self.lexer.get_cur_token() != Token.Number:
            return self.parse_error('number', 'in number')
        value = self.lexer.number_value
        self.lexer.consume(Token.Number)
        return NumberExprAST(loc, value)

    def parse_parentheses_expr(self):
        if self.lexer.get_cur_token() != Token.ParenthesesOpen:
            return self.parse_error('(', 'in parentheses')
        self.lexer.consume(Token.ParenthesesOpen)
        exp = self.parse_expression()
        if not exp:
            return None
        if self.lexer.get_cur_token() != Token.ParenthesesClose:
            return self.parse_error(')', 'in parentheses')
        self.lexer.consume(Token.ParenthesesClose)
        return exp

    def parse_tensor_literal_expr(self) -> Optional[LiteralExprAST]:
        loc = self.lexer.location.copy()
        if self.lexer.get_cur_token() != Token.SBracketOpen:
            return self.parse_error('[', 'in tensor literal')
        self.lexer.consume(Token.SBracketOpen)
        values = []
        while self.lexer.get_cur_token() != Token.SBracketClose:
            if self.lexer.get_cur_token() == Token.Number:
                value = self.parse_number_expr()
                if not value:
                    return None
                values.append(value)
            elif self.lexer.get_cur_token() == Token.SBracketOpen:
                value = self.parse_tensor_literal_expr()
                if not value:
                    return None
                values.append(value)
            else:
                return self.parse_error('<num> or [', 'in tensor literal')

            if self.lexer.get_cur_token() not in (Token.Comma, Token.SBracketClose):
                return self.parse_error(', or ]', 'in tensor literal')
            if self.lexer.get_cur_token() == Token.Comma:
                self.lexer.consume(Token.Comma)

        self.lexer.consume(Token.SBracketClose)
        size = len(values)
        if size == 0:
            return self.parse_error('at least one value', 'in tensor literal')

        dims = [size]
        if not all(isinstance(value, NumberExprAST) for value in values):
            if any(value.dims != values[0].dims for value in values):
                return self.parse_error('same shape', 'in tensor literal')

            dims += values[0].dims

        return LiteralExprAST(loc, values, dims)

    def parse_primary(self):
        if self.lexer.get_cur_token() == Token.Identifier:
            return self.parse_identifier_expr()
        elif self.lexer.get_cur_token() == Token.Number:
            return self.parse_number_expr()
        elif self.lexer.get_cur_token() == Token.ParenthesesOpen:
            return self.parse_parentheses_expr()
        elif self.lexer.get_cur_token() == Token.SBracketOpen:
            return self.parse_tensor_literal_expr()
        else:
            return self.parse_error('primary', 'in primary')

    @staticmethod
    def binop_precedence(op_token: str) -> int:
        return {
            '-': 20,
            '+': 20,
            '*': 40
        }[op_token]

    def parse_binop_rhs(self, lhs: ExprAST) -> Optional[ExprAST]:
        if self.lexer.get_cur_token() not in (Token.Minus, Token.Plus, Token.Mul):
            return lhs

        op = self.lexer.get_cur_token()
        loc = self.lexer.location.copy()
        self.lexer.consume(op)
        op_str = op.value
        op_loc = loc

        rhs = self.parse_primary()

        if isinstance(lhs, BinaryExprAST):
            r_op_str = op_str
            r_op_loc = op_loc
            if self.binop_precedence(lhs.op) < self.binop_precedence(r_op_str):
                op_str = lhs.op
                op_loc = lhs.location
                mid = lhs.rhs
                lhs = lhs.lhs
                rhs = BinaryExprAST(r_op_loc, r_op_str, mid, rhs)

        ret = BinaryExprAST(op_loc, op_str, lhs, rhs)
        return self.parse_binop_rhs(ret)

    def parse_expression(self) -> Optional[ExprAST]:
        lhs = self.parse_primary()
        if not lhs:
            return None
        return self.parse_binop_rhs(lhs)

    '''
    type ::= < shape_list >
    shape_list ::= num | num, shape_list
    '''
    def parse_type(self):
        if self.lexer.get_cur_token() != Token.Lt:
            return self.parse_error('<', 'in type')
        self.lexer.consume(Token.Lt)

        type_list = []
        while self.lexer.get_cur_token() == Token.Number:
            type_list.append(self.lexer.number_value)
            self.lexer.consume(Token.Number)
            if self.lexer.get_cur_token() != Token.Comma:
                break
            self.lexer.consume(Token.Comma)

        if self.lexer.get_cur_token() != Token.Gt:
            return self.parse_error('>', 'in type')
        self.lexer.consume(Token.Gt)
        return type_list

    def parse_declaration(self):
        loc = self.lexer.location.copy()
        if self.lexer.get_cur_token() != Token.Var:
            return self.parse_error('var', 'in declaration')
        self.lexer.consume(Token.Var)

        if self.lexer.get_cur_token() != Token.Identifier:
            return self.parse_error('variable name', 'in declaration')
        name = self.lexer.identifier
        self.lexer.consume(Token.Identifier)

        type_list = []
        if self.lexer.get_cur_token() == Token.Lt:
            type_list = self.parse_type()
            if type_list is None:
                return None

        if self.lexer.get_cur_token() != Token.Eq:
            return self.parse_error('=', 'in declaration')
        self.lexer.consume(Token.Eq)

        init_value = self.parse_expression()
        if init_value is None:
            return None

        return VarDeclExprAST(loc, name, VarType(type_list), init_value)

    def parse_prototype(self):
        loc = self.lexer.location.copy()

        if self.lexer.get_cur_token() != Token.Def:
            return self.parse_error('def', 'in prototype')
        self.lexer.consume(Token.Def)

        if self.lexer.get_cur_token() != Token.Identifier:
            return self.parse_error('function name', 'in prototype')
        name = self.lexer.identifier
        self.lexer.consume(Token.Identifier)

        if self.lexer.get_cur_token() != Token.ParenthesesOpen:
            return self.parse_error('(', 'in prototype')
        self.lexer.consume(Token.ParenthesesOpen)

        args = []
        while self.lexer.get_cur_token() != Token.ParenthesesClose:
            if self.lexer.get_cur_token() != Token.Identifier:
                return self.parse_error('argument name', 'in prototype arguments')
            args.append(VariableExprAST(self.lexer.location, self.lexer.identifier))
            self.lexer.consume(Token.Identifier)

            if self.lexer.get_cur_token() not in (Token.Comma, Token.ParenthesesClose):
                return self.parse_error(', or )', 'in prototype arguments')
            if self.lexer.get_cur_token() == Token.Comma:
                self.lexer.consume(Token.Comma)
        if self.lexer.get_cur_token() != Token.ParenthesesClose:
            return self.parse_error(')', 'in prototype arguments')
        self.lexer.consume(Token.ParenthesesClose)

        return PrototypeAST(loc, name, args)

    '''
    block ::= {expression_list}
    expression_list ::= block_expr;
    expression_list
    block_expr ::= decl | "return" | expr
    '''
    def parse_block(self):
        if self.lexer.get_cur_token() != Token.BraceOpen:
            return self.parse_error('{', 'in block')
        self.lexer.consume(Token.BraceOpen)

        block: ExprASTList = []
        while self.lexer.get_cur_token() == Token.Semicolon:
            self.lexer.consume(Token.Semicolon)

        while self.lexer.get_cur_token() != Token.BraceClose:
            if self.lexer.get_cur_token() == Token.Var:
                decl = self.parse_declaration()
                if decl is None:
                    return None
                block.append(decl)
            elif self.lexer.get_cur_token() == Token.Return:
                return_expr = self.parse_return()
                if return_expr is None:
                    return None
                block.append(return_expr)
            else:
                expr = self.parse_expression()
                if expr is None:
                    return None
                block.append(expr)

            if self.lexer.get_cur_token() != Token.Semicolon:
                return self.parse_error(';', 'in block')
            while self.lexer.get_cur_token() == Token.Semicolon:
                self.lexer.consume(Token.Semicolon)

        if self.lexer.get_cur_token() != Token.BraceClose:
            return self.parse_error('}', 'in block')
        self.lexer.consume(Token.BraceClose)
        return block

    def parse_definition(self):
        loc = self.lexer.location.copy()
        proto = self.parse_prototype()
        if proto is None:
            return None

        block = self.parse_block()
        if block is None:
            return None
        return FunctionAST(loc, proto, block)

    def parse_module(self):
        functions = []

        while self.lexer.get_cur_token() != Token.EOF:
            function = self.parse_definition()
            if function is None:
                return None
            functions.append(function)

        if self.lexer.get_cur_token() != Token.EOF:
            return self.parse_error('end of file', 'at end of module')
        self.lexer.consume(Token.EOF)

        return ModuleAST(self.lexer.location, functions)


