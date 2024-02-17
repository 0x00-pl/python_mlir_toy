import enum


class Location:
    def __init__(self, filename: str, line: int, column: int):
        self.filename = filename
        self.line = line
        self.column = column

    def dump(self):
        print(f'@{self.filename}:{self.line}:{self.column}', end='')


class Token(enum.Enum):
    Semicolon = ';'
    ParenthesesOpen = '('
    ParenthesesClose = ')'
    BraceOpen = '{'
    BraceClose = '}'
    SBracketOpen = '['
    SBracketClose = ']'
    EOF = 'EOF'
    Return = 'return'
    Var = 'var'
    Def = 'def'
    Identifier = 'identifier'
    Number = 'number'


class Lexer:
    def __init__(self, filename: str):
        self.location = Location(filename, 0, 0)
        self.cur_token: Token = Token.EOF
        self.identifier = ''
        self.number_value = 0
        self.line_buffer = ''
        self.cur_line_num = 0
        self.cur_column = 0
        self.last_char = ' '

    def read_next_line(self):
        raise NotImplementedError()

    def get_next_char(self):
        if not self.line_buffer:
            return Token.EOF

        self.cur_column += 1
        next_char = self.line_buffer[0]
        self.line_buffer = self.line_buffer[1:]
        if not self.line_buffer:
            self.line_buffer = self.read_next_line()
        if next_char == '\n':
            self.cur_line_num += 1
            self.cur_column = 0
        return next_char

    def get_token(self):
        while self.last_char.isspace():
            self.last_char = self.get_next_char()

        self.location.line = self.cur_line_num
        self.location.column = self.cur_column

        if self.last_char.isalpha():
            identifier = self.last_char
            while self.last_char.isalnum() or self.last_char == '_':
                self.last_char = self.get_next_char()
                identifier += self.last_char
            if identifier == 'return':
                self.cur_token = Token.Return
            elif identifier == 'var':
                self.cur_token = Token.Var
            elif identifier == 'def':
                self.cur_token = Token.Def
            else:
                self.cur_token = Token.Identifier
            self.identifier = identifier
            return self.cur_token

        if self.last_char.isnumeric() or self.last_char == '.':
            number_str = self.last_char
            while self.last_char.isnumeric() or self.last_char == '.':
                self.last_char = self.get_next_char()
                number_str += self.last_char
            self.number_value = float(number_str)
            return Token.Number

        if self.last_char == '#':
            while self.last_char not in (Token.EOF, '\n'):
                self.last_char = self.get_next_char()
            if self.last_char is not Token.EOF:
                return self.get_token()

        if self.last_char is Token.EOF:
            return Token.EOF

        token = Token(self.last_char)
        self.last_char = self.get_next_char()
        return token


class LexerBuffer(Lexer):
    def __init__(self, filename: str, buffer: str):
        super().__init__(filename)
        self.line_buffer = buffer
        self.lines = buffer.splitlines()
        self.current_line_index = 0

    def read_next_line(self):
        ret = ''
        if self.current_line_index < len(self.lines):
            ret = self.lines[self.current_line_index]
            self.current_line_index += 1

        return ret
