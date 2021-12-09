import re
from enum import Enum
import typing as t
from dataclasses import dataclass


@dataclass
class Unit:
    text: str
    type: t.Optional[Enum] = None
    def __repr__(self):
        return f'{self.type and self.type.name or self.type}({self.text!r})'


@dataclass
class LexError(Exception):
    token: str
    i: int = 0


@dataclass
class Lexer:

    patterns: t.Dict[str, str]
    symbols: t.Type[Enum]

    def __init__(self):
        self.patterns = {
            key: value.value
            for key, value in vars(self.symbols).items()
            if  key[0] != '_'
            and key[0] == key[0].upper()
        }

    def lex(self, tokens: t.Iterable[str], strict: bool = False) -> t.Iterable[Unit]:
        for i, token in enumerate(tokens):
            for key, pattern in self.patterns.items():
                if re.match(pattern, token):
                    yield Unit(token, getattr(self.symbols, key))
                    break
            else:
                if strict:
                    raise LexError(token, i)
                yield Unit(token)

    def lex_once(self, token: str, strict: bool = False) -> Unit:
        results = list(self.lex([token], strict=strict))
        return results[0]


class Symbols(Enum):
    Eq = '=='
    NONE = 'null'
    FALSE = 'false'
    TRUE = 'true'
    Ident = r'[\w_.]+'
    Dot = r'\.'
    StringLiteral = "^[\"']"
    LeftBracket, RightBracket = '{', '}'
    LeftPart, RightPart = '(', ')'
    LeftSquareBracket, RightSquareBracket = '[', ']'


class MyLexer(Lexer):
    symbols = Symbols


def test_init():
    class aLexer(Lexer):
        class symbols(Enum):
            A = 'a'
            B = 'b'
    
    assert aLexer().patterns == {
        'A': 'a', 'B': 'b'
    }


def test_basic():
    
    ml = MyLexer()
    print(ml, ml.symbols)
    result = list(ml.lex(['a', '==', 'b']))
    print(result)
    assert result == [
        Unit('a', Symbols.Ident),
        Unit('==',Symbols.Eq),
        Unit('b', Symbols.Ident)
    ]
    assert list(ml.lex(['"a"'])) == [Unit('"a"', Symbols.StringLiteral)]
