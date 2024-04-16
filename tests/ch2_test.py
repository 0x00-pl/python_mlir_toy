import pytest

from python_mlir_toy.ch2 import toy


def test_help_info():
    with pytest.raises(SystemExit):
        toy.main(['-h'])


def test_convert_toy_to_ast():
    toy.main(['tests/transpose.toy', '-emit=ast'])


def test_convert_toy_to_mlir():
    toy.main(['tests/transpose.toy', '-emit=mlir'])


def test_convert_mlir_to_ast():
    with pytest.raises(AssertionError):
        toy.main(['tests/transpose.mlir', '-emit=ast'])


def test_convert_mlir_to_mlir():
    toy.main(['tests/transpose.mlir', '-emit=mlir'])


if __name__ == '__main__':
    test_help_info()
    test_convert_toy_to_ast()
    test_convert_toy_to_mlir()
    test_convert_mlir_to_ast()
    test_convert_mlir_to_mlir()
