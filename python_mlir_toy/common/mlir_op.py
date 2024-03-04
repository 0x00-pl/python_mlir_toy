from python_mlir_toy.common import location


class ModuleOp:
    def __init__(self, loc: location.Location, name=None):
        self.loc = loc
        self.name = name
