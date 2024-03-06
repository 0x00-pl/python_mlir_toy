from typing import List, Dict

from python_mlir_toy.common import td


class SymbolTable:
    def __init__(self):
        self.scope: List[Dict[str, td.Value]] = []

    def insert(self, name: str, value: td.Value):
        self.scope[-1][name] = value

    def lookup(self, name: str):
        for scope in reversed(self.scope):
            if name in scope:
                return scope[name]
        return None

    def __enter__(self):
        self.scope.append({})

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.scope.pop()
