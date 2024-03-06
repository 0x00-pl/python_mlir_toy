import typing

from python_mlir_toy.common import location, td


class ModuleOp(td.Op):
    def __init__(self, loc: location.Location, name: str, func_dict: typing.Dict[str, td.Op]):
        super().__init__(loc, 'module')
        self.name = name
        self.func_dict = func_dict
