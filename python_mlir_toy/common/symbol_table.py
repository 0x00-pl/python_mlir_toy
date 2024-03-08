from typing import List, Dict


class SymbolTable:
    def __init__(self):
        self.scope: List[Dict[str, 'td.value']] = [{}]

    def insert(self, name: str, value: 'td.Value'):
        self.scope[-1][name] = value

    def lookup(self, name: str):
        for scope in reversed(self.scope):
            if name in scope:
                return scope[name]
        return None

    def next_unused_symbol(self, prefix: str = '%'):
        index = 0
        while True:
            name = f'{prefix}{index}'
            if self.lookup(name) is None:
                return name
            index += 1

    def __enter__(self):
        self.scope.append({})

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.scope.pop()
