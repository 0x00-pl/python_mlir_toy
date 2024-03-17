from python_mlir_toy.common import serializable, scoped


class MlirParser(serializable.TextParser, scoped.Scoped):
    def parse(self):
        raise NotImplementedError
