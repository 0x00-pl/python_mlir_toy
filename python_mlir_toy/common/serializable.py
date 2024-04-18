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
        self._line_buffer = self.file.readline()
        self._cur_char = self._line_buffer[self.cur_pose]
        self._last_token = None
        self._last_token_kind: TokenKind = TokenKind.Unknown
        self.drop_token()
        self._number_pattern = re.compile(r'[-+]?(([0-9]*\.?[0-9]*([eE][-+]?[0-9]+)?)|0[xo][0-9a-fA-F]+)')

    def get_location(self):
        return self.filename, self.cur_line + 1, self.cur_pose + 1

    def cur_char(self) -> str:
        return self._cur_char

    # def last_char(self) -> str:
    #     return self._last_char
    def drop_line(self):
        self.cur_line += 1
        self.cur_pose = 0
        self._line_buffer = self.file.readline()

    def drop_char(self, check_char: str = None):
        if check_char is not None:
            assert self._cur_char == check_char

        self.cur_pose += 1
        if self.cur_pose >= len(self._line_buffer):
            self.drop_line()

        if len(self._line_buffer) != 0:
            self._cur_char = self._line_buffer[self.cur_pose]
        else:
            self._cur_char = None

    def is_space_or_comment(self):
        is_space = self.cur_char() is not None and self.cur_char().isspace()
        is_comment = self.cur_char() == '/' and self.cur_pose + 1 < len(self._line_buffer) and self._line_buffer[
            self.cur_pose + 1] == '/'
        return is_space or is_comment

    def drop_space(self):
        while self.is_space_or_comment():
            if self.cur_char() == '/':
                self.drop_line()
            else:
                self.drop_char()

    def last_token(self):
        return self._last_token

    def last_token_kind(self):
        return self._last_token_kind

    def drop_token(self, check_token: str = None, check_kind: TokenKind = None, skip_space: bool = True):
        if check_token is not None:
            assert self.last_token() == check_token

        if check_kind is not None:
            assert self.last_token_kind() == check_kind

        if skip_space:
            self.drop_space()

        if self.cur_char() is None:
            self._last_token, self._last_token_kind = None, TokenKind.EOF
        elif self.cur_char().isidentifier():
            identifier = ''
            while self.cur_char() is not None and (identifier + self.cur_char()).isidentifier():
                identifier += self.cur_char()
                self.drop_char()
            self._last_token, self._last_token_kind = identifier, TokenKind.Identifier
        elif self.cur_char().isdigit():
            number_str = ''
            while self.cur_char() is not None and self._number_pattern.fullmatch(
                    number_str + self.cur_char()) is not None:
                number_str += self.cur_char()
                self.drop_char()
            if '.' in number_str:
                self._last_token, self._last_token_kind = float(number_str), TokenKind.Number
            else:
                self._last_token, self._last_token_kind = int(number_str), TokenKind.Number
        elif self.cur_char() == '"':
            self.drop_char()
            string = ''
            while self.cur_char() is not None and self.cur_char() != '"':
                if self.cur_char() == '\\':
                    self.drop_char()
                string += self.cur_char()
                self.drop_char()
            self.drop_char('"')
            self._last_token, self._last_token_kind = string, TokenKind.String
        else:
            last_char = self.cur_char()
            self.drop_char()
            self._last_token, self._last_token_kind = last_char, TokenKind.Other


class Serializable:
    pass


class TextSerializable(Serializable):
    def print(self, dst: TextPrinter):
        raise NotImplementedError

    @classmethod
    def parse(cls, src: TextParser):
        raise NotImplementedError

    def dump(self):
        printer = TextPrinter(file=sys.stdout)
        self.print(printer)

    def __str__(self):
        file = io.StringIO()
        printer = TextPrinter(file=file)
        self.print(printer)
        return file.getvalue()


class Empty(TextSerializable):
    def print(self, dst: TextPrinter):
        pass

    @classmethod
    def parse(cls, src: TextParser):
        pass


empty = Empty()
