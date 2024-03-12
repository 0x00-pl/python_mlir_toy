import typing


class Scoped:
    def __enter__(self):
        raise NotImplementedError(f'{type(self)} does not implement __enter__()')

    def __exit__(self, exc_type, exc_val, exc_tb):
        raise NotImplementedError(f'{type(self)} does not implement __exit__()')


class Indent(Scoped):
    def __init__(self, level: int = 0):
        super().__init__()
        self.level = level

    def __enter__(self):
        self.level += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.level -= 1

    def __str__(self):
        return '  ' * self.level

    def dump(self):
        print(self, end='')


K = typing.TypeVar('K')
V = typing.TypeVar('V')


class KVScoped(Scoped, typing.Generic[K, V]):
    def __init__(self):
        self.stack: typing.List[typing.Dict[K, V]] = [{}]

    def insert(self, key: K, value: V):
        self.stack[-1][key] = value

    def lookup(self, key: K) -> typing.Optional[V]:
        for scope in reversed(self.stack):
            if key in scope:
                return scope[key]
        return None

    def __enter__(self):
        self.stack.append({})

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stack.pop()


class SymbolTable(KVScoped[str, V], typing.Generic[V]):
    def __init__(self):
        super().__init__()

    def next_unused_symbol(self, prefix: str = '%'):
        index = 0
        while True:
            name = f'{prefix}{index}'
            if self.lookup(name) is None:
                return name
            index += 1
