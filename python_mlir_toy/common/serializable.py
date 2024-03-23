import enum
import io
import re
import sys
import typing


class TextPrinter:
    def __init__(self, sep=' ', end=' ', file: typing.TextIO = sys.stdout):
        self.sep = sep
        self.end = end
        self.file = file

    def print(self, *values, sep=None, end=None, flush=False):
        sep = sep if sep is not None else self.sep
        end = end if end is not None else self.end
        print(*values, sep=sep, end=end, file=self.file, flush=flush)

    def print_newline(self):
        self.print(end='\n')


class TemplatedSymbol:
    def __init__(self, identifier_list: typing.List[str], template_list: typing.List['TemplatedSymbol']):
        self.identifier_list = identifier_list
        self.template_list = template_list

    def __str__(self):
        ret = f'{".".join(self.identifier_list)}'
        if len(self.template_list) != 0:
            ret += f'<{", ".join(str(i) for i in self.template_list)}>'
        return ret


class TemplatedSymbolParser:
    def __init__(self, templated_symbol_str: str):
        self.templated_symbol_str = templated_symbol_str
        self.position = 0
        self.last_char = templated_symbol_str[0] if templated_symbol_str else None
        self.last_token = None
        self.move_next_token()

    def get_char(self) -> str:
        return self.last_char

    def move_next_char(self) -> None:
        self.position += 1
        if self.position < len(self.templated_symbol_str):
            self.last_char = self.templated_symbol_str[self.position]
        else:
            self.last_char = None
            self.position = len(self.templated_symbol_str)

    def skip_space(self) -> None:
        while self.get_char() is not None and self.get_char().isspace():
            self.move_next_char()

    def get_token(self):
        return self.last_token

    def move_next_token(self):
        self.skip_space()
        if self.get_char() is None:
            self.last_token = None
        elif self.get_char() in ',.<>':
            self.last_token = self.get_char()
            self.move_next_char()
        elif self.get_char().isidentifier():
            identifier = ''
            while self.get_char() is not None and (identifier + self.get_char()).isidentifier():
                identifier += self.get_char()
                self.move_next_char()
            self.last_token = identifier
        else:
            raise ValueError(f'Unknown character: "{self.get_char()}" in {self.templated_symbol_str}[{self.position}]')

    def parse(self) -> TemplatedSymbol:
        identifier_list = []
        template_list = []

        while self.get_token() is not None and self.get_token() not in '<>,':
            if self.get_token() == '.':
                pass
            elif self.get_token() == '<':
                break
            else:
                identifier_list.append(self.get_token())
            self.move_next_token()

        if self.get_token() == '<':
            self.move_next_token()
            while self.get_token() != '>':
                template_list.append(self.parse())
                assert self.get_token() in ',>'
                if self.get_token() == ',':
                    self.move_next_token()

            assert self.get_token() == '>'
            self.move_next_token()

        return TemplatedSymbol(identifier_list, template_list)

    def __str__(self):
        return str(self.parse())


class TokenKind(enum.Enum):
    Space = 0
    Comment = 1
    Number = 2
    Identifier = 3
    String = 4
    Other = 5
    Error = 6
    EOF = 7
    Unknown = 8


class TextParser:

    def __init__(self, file: typing.TextIO = sys.stdin, filename: str = 'unknown'):
        self.file = file
        self.filename = filename
        self.cur_line = 0
        self.cur_pose = 0
        self._last_location = None
        self._line_buffer = self.file.readline()
        self._last_char = None
        self._last_token = None
        self._last_token_kind: TokenKind = TokenKind.Unknown
        self.process_char()
        self.process_token()
        self._number_pattern = re.compile(r'[-+]?(([0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?)|0[xo][0-9a-fA-F]+)')

    def set_last_location(self):
        self._last_location = (self.filename, self.cur_line + 1, self.cur_pose + 1)

    def last_location(self):
        return self._last_location

    def last_char(self) -> str:
        return self._last_char

    def process_char(self, check_char: str = None):
        if check_char is not None:
            assert self.last_char() == check_char

        self.set_last_location()
        self._last_char = self._line_buffer[self.cur_pose]

        self.cur_pose += 1
        if self.cur_pose >= len(self._line_buffer):
            self.cur_line += 1
            self._line_buffer = self.file.readline()
            self.cur_pose = 0

    def process_space(self, check_space: bool = False):
        if check_space:
            assert self.last_char().isspace()

        self.set_last_location()
        while self.last_char() is not None and (self.last_char().isspace() or self.last_char() == '/'):
            prev_char = self.last_char()
            self.process_char()
            if prev_char == '/' and self.last_char() == '/':  # //comment
                while self.last_char() is not None and self.last_char() != '\n':
                    self.process_char()

    def last_token(self):
        return self._last_token

    def last_token_kind(self):
        return self._last_token_kind

    def process_token(self, check_token: str = None, check_kind: TokenKind = None):
        if check_token is not None:
            assert self.last_token() == check_token

        if check_kind is not None:
            assert self.last_token_kind() == check_kind

        self.process_space()

        self.set_last_location()
        while self.last_char() is not None:
            if self.last_char().isidentifier():
                identifier = ''
                while self.last_char().isidentifier():
                    identifier += self.last_char()
                    self.process_char()
                self._last_token, self._last_token_kind = identifier, TokenKind.Identifier
                return
            elif self.last_char().isdigit():
                number_str = ''
                while self._number_pattern.match(number_str + self.last_char()) is not None:
                    number_str += self.last_char()
                    self.process_char()
                if '.' in number_str:
                    self._last_token, self._last_token_kind = float(number_str), TokenKind.Number
                else:
                    self._last_token, self._last_token_kind = int(number_str), TokenKind.Number
                return
            # elif self.last_char() == '/':
            #     self.process_char()
            #     if self.last_char() == '/':
            #         self.process_char()
            #         content = ''
            #         while self.last_char() != '\n':
            #             content += self.last_char()
            #             self.process_char()
            #         self._last_token, self._last_token_kind = content, TokenKind.Comment
            #     else:
            #         self._last_token, self._last_token_kind = '/', TokenKind.Other
            #     return
            elif self.last_char() == '"':
                self.process_char()
                string = ''
                while self.last_char() != '"':
                    if self.last_char() == '\\':
                        self.process_char()
                    if self.last_char() is None:
                        raise ValueError(f'Unterminated string in {self.filename}[{self.cur_line + 1}:{self.cur_pose + 1}]')
                    string += self.last_char()
                    self.process_char()
                self.process_char()
                self._last_token, self._last_token_kind = string, TokenKind.String
                return
            else:
                self._last_token, self._last_token_kind = self.last_char(), TokenKind.Other
                self.process_char()
                return
        self._last_token, self._last_token_kind = None, TokenKind.EOF


class Serializable:
    pass


class TextSerializable(Serializable):
    def print(self, dst: TextPrinter):
        raise NotImplementedError

    def dump(self):
        printer = TextPrinter(file=sys.stdout)
        self.print(printer)

    def __str__(self):
        file = io.StringIO()
        printer = TextPrinter(file=file)
        self.print(printer)
        return file.getvalue()

    @classmethod
    def parse(cls, src: TextParser):
        raise NotImplementedError


class Empty(TextSerializable):
    def print(self, dst: TextPrinter):
        pass

    @classmethod
    def parse(cls, src: TextParser):
        pass


empty = Empty()
