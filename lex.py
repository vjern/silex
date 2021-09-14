import re
import enum
import typing as t
from dataclasses import dataclass


@dataclass
class Unit:
    text: str
    type: str = None


@dataclass
class Lexer:
    patterns: t.Dict[str, str]
    Symbols: type
    def __init__(self):
        self.patterns = {
            key: value
            for key, value in vars(self.__class__).items()
            if key[0] != '_'
            if key[0] == key[0].upper()
        }
        self.Symbols = enum.Enum('Symbols', list(self.patterns.keys()))
        
    def lex(self, tokens: t.List[str]) -> t.List[Unit]:
        for token in tokens:
            for key, pattern in self.patterns.items():
                if re.match(pattern, token):
                    yield Unit(token, getattr(self.Symbols, key))
                    break
            else:
                yield Unit(token)


class MyLexer(Lexer):
    EQ = '=='
    NONE = 'none'
    NIL = 'nil'
    FALSE = 'false'
    TRUE = 'true'
    IDENT = r'[\w_.]+'
    DOT = r'\.'
    STRING_LITERAL = "^[\"']"


def test_init():
    class aLexer(Lexer):
        A = 'a'
        B = 'b'
    
    assert aLexer().patterns == {
        'A': 'a', 'B': 'b'
    }

def test_basic():
    
    ml = MyLexer()
    print(ml, ml.Symbols.__members__)
    assert list(ml.lex(['a', '==', 'b'])) == [
        Unit('a', 'IDENT'),
        Unit('==', 'EQ'),
        Unit('b', 'IDENT')
    ]
    assert list(ml.lex(['"a"'])) == [Unit('"a"', 'STRING_LITERAL')]
    assert 0
