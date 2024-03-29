import typing

from python_mlir_toy.common import serializable, tools


class Format:
    def print(self, obj, dst: serializable.TextPrinter):
        obj.print(dst)

    def parse(self, src: serializable.TextParser):
        return None


class StrFormat(Format):
    def print(self, obj, dst: serializable.TextPrinter):
        assert isinstance(obj, str)
        dst.print(obj)

    def parse(self, src: serializable.TextParser):
        ret = src.last_token()
        src.process_token()
        return ret


class NamespacedSymbolFormat(Format):
    def print(self, obj, dst: serializable.TextPrinter):
        assert isinstance(obj, str)
        dst.print(obj)

    def parse(self, src: serializable.TextParser):
        symbol_name = src.last_token()
        src.process_token(check_kind=serializable.TokenKind.Identifier, skip_space=False)
        while src.last_token() == '.':
            src.process_token(skip_space=False)
            symbol_name += '.' + src.last_token()
            src.process_token(check_kind=serializable.TokenKind.Identifier, skip_space=False)
        return symbol_name


class VariableNameFormat(Format):
    def __init__(self, prefix: str):
        self.prefix = prefix

    def print(self, obj, dst: serializable.TextPrinter):
        assert isinstance(obj, str)
        assert obj.startswith(self.prefix)
        dst.print(obj)

    def parse(self, src: serializable.TextParser):
        src.process_token(self.prefix, skip_space=False)
        variable_name = src.last_token()
        src.process_token()
        return self.prefix + variable_name


class ConstantStrFormat(Format):
    def __init__(self, text: str):
        self.text = text

    def print(self, obj, dst: serializable.TextPrinter):
        dst.print(self.text)

    def parse(self, src: serializable.TextParser):
        src.process_token(self.text.strip())


class ListFormat(Format):
    def __init__(self, content_format_list: typing.List[Format]):
        self.content_format_list = content_format_list

    def print(self, obj, dst: serializable.TextPrinter):
        for content_format in self.content_format_list:
            content_format.print(obj, dst)

    def parse(self, src: serializable.TextParser):
        ret = []
        for content_format in self.content_format_list:
            ret.append(content_format.parse(src))
        return ret


class RepeatFormat(Format):
    def __init__(self, content_format: Format, sep_format: str | ConstantStrFormat = ','):
        self.content_format = content_format
        self.sep_format = sep_format if isinstance(sep_format, ConstantStrFormat) else ConstantStrFormat(sep_format)

    def print(self, obj, dst: serializable.TextPrinter):
        for content in tools.with_sep(obj, lambda: self.sep_format.print(serializable.empty, dst)):
            self.content_format.print(content, dst)

    def parse(self, src: serializable.TextParser) -> typing.List[typing.Any]:
        ret = []
        while src.last_token() is not self.sep_format.text.strip():
            ret.append(self.content_format.parse(src))
            ret.append(self.sep_format.parse(src))
        return ret


class DictFormat(RepeatFormat):
    def __init__(self, k_format: Format, v_format: Format,
                 kv_sep: str | ConstantStrFormat = ':', item_sep: str | ConstantStrFormat = ','):
        super().__init__(ListFormat([k_format, kv_sep, v_format]), item_sep)

    def print(self, obj, dst: serializable.TextPrinter):
        super().print(obj.items(), dst)

    def parse(self, src: serializable.TextParser):
        return {item[0]: item[2] for item in super().parse(src)}


class OptionalFormat(Format):
    def __init__(self, content_format: Format, pred: typing.Callable[[typing.Any], bool], prefix: str):
        self.content_format = content_format
        self.pred = pred
        self.prefix = prefix

    def print(self, obj, dst: serializable.TextPrinter):
        if self.pred(obj):
            self.content_format.print(obj, dst)

    def parse(self, src: serializable.TextParser):
        if src.last_token() == self.prefix:
            return self.content_format.parse(src)
        return None
