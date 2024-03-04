import argparse
import enum

from python_mlir_toy.ch2 import ast
from python_mlir_toy.ch2.lexer import LexerBuffer
from python_mlir_toy.ch2.parser import Parser


class Action(enum.Enum):
    Ast = 'ast'
    Mlir = 'mlir'


def build_arg_parser():
    arg_parser = argparse.ArgumentParser('toy compiler')
    arg_parser.add_argument('input_file', nargs='?', type=argparse.FileType('r'), default='-', help='input toy file')
    arg_parser.add_argument('-emit', dest='emit_action', nargs=1, type=str, choices=[i.value for i in Action],
                            help=f'Select the kind of output desired: {Action.Ast}(output the AST dump)')
    return arg_parser


def dump_ast(args):
    lexer = LexerBuffer(args.input_file.name, args.input_file)
    parser = Parser(lexer)
    module_ast = parser.parse_module()
    ast.dump(module_ast)


def main(argv=None):
    arg_parser = build_arg_parser()
    args = arg_parser.parse_args(argv)

    arg_action = Action(args.emit_action[0])
    if arg_action == Action.Ast:
        dump_ast(args)
    elif arg_action == Action.Mlir:
        raise NotImplementedError('MLIR generation not yet implemented')
    else:
        raise 'No action specified (parsing only?), use -emit=<action>'


if __name__ == '__main__':
    main(['tests/transpose.mlir', '-emit=ast', '-emit=ast'])
