

class Location:
    def __init__(self, filename: str, line: int, column: int):
        self.filename = filename
        self.line = line
        self.column = column
