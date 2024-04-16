import pytest

from python_mlir_toy.ch1 import toy


def test_help_info():
    with pytest.raises(SystemExit):
        toy.main(['-h'])


def test_convert_toy_to_ast():
    toy.main(['tests/transpose.toy', '-emit=ast'])


if __name__ == '__main__':
    test_help_info()
    test_convert_toy_to_ast()
