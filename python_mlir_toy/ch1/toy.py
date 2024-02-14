import argparse
import enum

from python_mlir_toy.ch1 import ast


class Action(enum.Enum):
    Ast = 'ast'


def build_arg_parser():
    arg_parser = argparse.ArgumentParser('toy compiler')
    arg_parser.add_argument('filename', nargs='?', type=argparse.FileType('r'), help='input toy file')
    arg_parser.add_argument('-emit', dest='emitAction', nargs=1, type=str, choices=[i.value for i in Action],
                            help=f'Select the kind of output desired: {Action.Ast}(output the AST dump)')
    return arg_parser


def main(argv=None):
    arg_parser = build_arg_parser()
    args = arg_parser.parse_args(argv)

    moduleAST = None  # TODO

    arg_action = Action(args.emitAction[0])
    if arg_action == Action.Ast:
        ast.dump(moduleAST)
    else:
        raise 'No action specified (parsing only?), use -emit=<action>'


if __name__ == '__main__':
    main(['tests/main.mlir', '-emit=ast', '-emit=ast'])
