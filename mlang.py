from enum import Enum

import g
import lex


class Symbols(g.Symbols, Enum):
    Equal = '='
    Star = '*'
    NumberLiteral = r'\d+'
    Alteration = r'#|b'
    Slash = '/'


class Rhythm:
    _slash: Symbols.Slash
    value: g.U[Symbols.Ident, Symbols.NumberLiteral]


class Alteration:
    value: Symbols.Alteration


class Multiply:
    _: Symbols.Star
    value: Symbols.NumberLiteral


class Note:
    note: Symbols.Ident
    octave: Optional[Symbols.NumberLiteral]
    alt: Optional[Alteration]
    rhythm: Optional[Rhythm]
    mult: Optional[Multiply]


class Expr(g.U):
    candidates = [
        Note,
        Symbols.Ident,
        Block,
    ]


class Block(g.ASynt):
    _start: Symbols.LeftBracket
    content: g.Chain[Expr]
    _end: Symbols.RightBracket


class Assignment(g.ASynt):
    name: Symbols.Ident
    _eq: Symbols.Equal
    value: Block


class Instruction(g.SyntUnion):
    candidates = [
        Expr,
        Assignment
    ]


class Program(g.ASynt):
    instructions: g.Chain[Instruction]


def test_a():

    units = [
        lex.Unit('a', Symbols.Ident)
    ]

    assert Program.parse(units) == (
        Program(instructions=[
            Note(note=lex.Unit('a', Symbols.Ident))
        ]),
        1
    )


def test_b():

    units = [
        lex.Unit('{', Symbols.LeftBracket),
        lex.Unit('}', Symbols.RightBracket)
    ]

    assert Program.parse(units) == (
        Program(instructions=[
            Block(content=[])
        ]),
        2
    )
