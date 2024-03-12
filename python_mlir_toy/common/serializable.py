import io
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


class TextParser:
    def __init__(self, file: typing.TextIO = sys.stdin, filename: str = 'unknown'):
        self.file = file
        self.filename = filename
        self.cur_line = -1
        self.cur_pose = -1
        self.line_buffer = ''
        self.cur_char_cache = None

    def cur_location(self):
        return self.filename, self.cur_line + 1, self.cur_pose + 1

    def cur_char(self):
        if self.cur_char_cache is None:
            self.move_next_char()
        return self.cur_char_cache

    def move_next_char(self):
        self.cur_pose += 1
        if self.cur_pose >= len(self.line_buffer):
            self.cur_line += 1
            self.line_buffer = self.file.readline()
            self.cur_pose = 0
        self.cur_char_cache = self.line_buffer[self.cur_pose]

    def move_skip_space(self):
        while self.cur_char().isspace():
            self.move_next_char()


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

    def parse(self, src: TextParser):
        raise NotImplementedError

    def consume(self, src: TextParser) -> bool:
        raise NotImplementedError
