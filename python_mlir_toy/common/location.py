class Location:
    pass


class UnknownLocation(Location):
    pass


class FileLineColLocation(Location):
    def __init__(self, filename, line, column):
        self.filename = filename
        self.line = line
        self.column = column

