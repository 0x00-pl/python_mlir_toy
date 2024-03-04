import enum


class Location:
    def __init__(self, filename: str, line: int, column: int):
        self.filename = filename
        self.line = line
        self.column = column

    def __str__(self):
        return f'@{self.filename}:{self.line}:{self.column}'

    def dump(self):
        print(str(self), end='')

    def copy(self):
        return Location(self.filename, self.line, self.column)


class Token(enum.Enum):
    Semicolon = ';'
    Comma = ','
    ParenthesesOpen = '('
    ParenthesesClose = ')'
    BraceOpen = '{'
    BraceClose = '}'
    SBracketOpen = '['
    SBracketClose = ']'
    Lt = '<'
    Gt = '>'
    Eq = '='
    Minus = '-'
    Plus = '+'
    Mul = '*'
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
        self.cur_line_num = 1
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

    def get_cur_token(self):
        return self.cur_token

    def get_next_token(self):
        while self.last_char != Token.EOF and self.last_char.isspace():
            self.last_char = self.get_next_char()

        if self.last_char is Token.EOF:
            self.cur_token = Token.EOF
            return self.cur_token

        self.location.line = self.cur_line_num
        self.location.column = self.cur_column

        if self.last_char.isalpha():
            identifier = ''
            while self.last_char.isalnum() or self.last_char == '_':
                identifier += self.last_char
                self.last_char = self.get_next_char()
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
            number_str = ''
            while self.last_char.isnumeric() or self.last_char == '.':
                number_str += self.last_char
                self.last_char = self.get_next_char()
            self.number_value = float(number_str)
            self.cur_token = Token.Number
            return self.cur_token

        if self.last_char == '#':
            while self.last_char not in (Token.EOF, '\n'):
                self.last_char = self.get_next_char()
            if self.last_char is not Token.EOF:
                return self.get_next_token()

        token = Token(self.last_char)
        self.last_char = self.get_next_char()
        self.cur_token = token
        return self.cur_token

    def consume(self, token: Token):
        if self.get_cur_token() != token:
            raise Exception(f'Expected {token}, got {self.get_cur_token()}')
        self.get_next_token()


class LexerBuffer(Lexer):
    def __init__(self, filename: str, buffer):
        super().__init__(filename)
        self.line_buffer = ''
        self.lines = buffer.readlines()
        self.current_line_index = 0
        self.line_buffer = self.read_next_line()
        self.get_next_token()

    def read_next_line(self):
        ret = ''
        if self.current_line_index < len(self.lines):
            ret = self.lines[self.current_line_index]
            self.current_line_index += 1

        return ret
