class Indent:
    def __init__(self, level=0):
        self.level = 0

    def __enter__(self):
        self.level += 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.level -= 1

    def __str__(self):
        return '  ' * self.level

    def dump(self):
        print(self, end='')
