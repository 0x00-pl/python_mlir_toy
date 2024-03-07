import argparse
import enum

from python_mlir_toy.ch1 import ast
from python_mlir_toy.ch1.parser import Parser
from python_mlir_toy.ch1.lexer import LexerBuffer


class Action(enum.Enum):
    Ast = 'ast'


def build_arg_parser():
    arg_parser = argparse.ArgumentParser('toy compiler')
    arg_parser.add_argument('input_file', nargs='?', type=argparse.FileType('r'), default='-', help='input toy file')
    arg_parser.add_argument('-emit', dest='emit_action', nargs=1, type=str, choices=[i.value for i in Action],
                            help=f'Select the kind of output desired: {Action.Ast}(output the AST dump)')
    return arg_parser


def main(argv=None):
    arg_parser = build_arg_parser()
    args = arg_parser.parse_args(argv)

    lexer = LexerBuffer(args.input_file.name, args.input_file)
    parser = Parser(lexer)
    module_ast = parser.parse_module()

    arg_action = Action(args.emit_action[0])
    if arg_action == Action.Ast:
        ast.dump(module_ast)
    else:
        raise 'No action specified (parsing only?), use -emit=<action>'


if __name__ == '__main__':
    main(['tests/transpose.mlir', '-emit=ast'])
