from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Any, Union

import lex


class Synt:
    value: Any

    def __init__(self, value = None, **kw):
        self.value = value
        for key, value in kw.items():
            setattr(self, key, value)

    @classmethod
    def parse(cls, units: List[lex.Unit]) -> Tuple[Union['Synt', lex.Unit], int]:
        if not units:
            raise ValueError('No units to parse!')
        return units[0], 1


class Symbols(Enum):
    Ident = 'ident'
    LeftPar, RightPar = '(', ')'
    LeftBracket, RightBracket = '{', '}'
    LeftSquareBracket, RightSquareBracket = '[', ']'
    FuncKeyword = 'fn'
    Comma = ','


@dataclass
class ItemList(Synt):
    left, right = Symbols.LeftPar, Symbols.RightPar
    delimiter = Symbols.Comma
    T = Synt

    value: List[Any]

    @classmethod
    def __class_getitem__(cls, target: type):
        return type(target.__name__ + 'List', (cls,), {'T': target, 'base': cls})

    @classmethod
    def parse(cls, units):

        skip = 2  # for left+right delimiters
        items = []
        # expect a left par
        assert units[0].type == cls.left
        units = units[1:]
        # now alternate between target and commma
        boat = False
        while units[0].type != cls.right:
            if boat:
                # expect a comma
                assert units[0].type == cls.delimiter, f'Expected {cls.delimiter!r}, got {units[0]!r}'
                units = units[1:]
                skip += 1
            else:
                # expect a TARGET
                obj, tskip = cls.T.parse(units)
                skip += tskip
                units = units[tskip:]
                items.append(obj)
            boat = not boat

        # expect a right par
        assert units[0].type == cls.right
        return getattr(cls, 'base', cls)(items), skip


class ASynt(Synt):

    def __repr__(self):
        astr = ', '.join('%s=%s' % (key, getattr(self, key)) for key in self.__annotations__)
        return '%s(%s)' % (self.__class__.__name__, astr)
    
    def __eq__(self, obj) -> bool:
        if not isinstance(obj, self.__class__):
            return False
        for key in self.__annotations__:
            if getattr(obj, key) != getattr(self, key):
                return False
        return True

    @classmethod
    def parse(cls, units):
        skip = 0
        data = {}
        for key, value in cls.__annotations__.items():
            if isinstance(value, Enum):
                obj = units[0]
                assert obj.type == value
                tskip = 1
            else:
                obj, tskip = value.parse(units)
            data[key] = obj
            skip += tskip
            units = units[tskip:]
        return cls(**data), skip


class TypeExpr(ASynt):
    value: Symbols.Ident


class Arg(ASynt):
    name: Symbols.Ident
    type: TypeExpr


class ArgList(ItemList):
    items: Arg


class TestASynt:

    def test_basic(self):

        class Arg(ASynt):
            a: Symbols.Ident
            b: Symbols.Ident

        units = [
            lex.Unit('a', Symbols.Ident),
            lex.Unit('b', Symbols.Ident)
        ]
        
        assert Arg.parse(units) == (
            Arg(
                a=lex.Unit('a', Symbols.Ident),
                b=lex.Unit('b', Symbols.Ident)
            ),
            2
        )
    
    def test_nested(self):

        class TypeExpr(ASynt):
            name: Symbols.Ident
        
        class Param(ASynt):
            name: Symbols.Ident
            type: TypeExpr

        class Func(ASynt):
            name: Symbols.Ident
            paramList: ItemList[Param]
            returnType: TypeExpr
        
        units = [
            lex.Unit('doThing', Symbols.Ident),
            lex.Unit('(', Symbols.LeftPar),
                lex.Unit('thing', Symbols.Ident), lex.Unit('int', Symbols.Ident),
            lex.Unit(')', Symbols.RightPar),
            lex.Unit('str', Symbols.Ident)
        ]

        result, offset = Func.parse(units)
        assert offset == 6
        assert result == Func(
            name=lex.Unit('doThing', Symbols.Ident),
            paramList=ItemList([
                Param(
                    name=lex.Unit('thing', Symbols.Ident),
                    type=TypeExpr(name=lex.Unit('int', Symbols.Ident))
                )
            ]),
            returnType=TypeExpr(name=lex.Unit('str', Symbols.Ident))
        )


class TestItemList:

    def test_basic(self):
        
        units = [
            lex.Unit('(', Symbols.LeftPar),
            lex.Unit('a', Symbols.Ident),
            lex.Unit(',', Symbols.Comma),
            lex.Unit('b', Symbols.Ident),
            lex.Unit(')', Symbols.RightPar)
        ]

        result, offset = ItemList.parse(units)
        assert offset == 5
        assert result == ItemList([
            lex.Unit('a', Symbols.Ident),
            lex.Unit('b', Symbols.Ident)
        ])

    def test_custom_brackets(self):
    
        class ItemSet(ItemList):
            left, right = Symbols.LeftBracket, Symbols.RightBracket
        
        units = [
            lex.Unit('(', Symbols.LeftBracket),
            lex.Unit('a', Symbols.Ident),
            lex.Unit(',', Symbols.Comma),
            lex.Unit('b', Symbols.Ident),
            lex.Unit(')', Symbols.RightBracket)
        ]

        result, offset = ItemSet.parse(units)
        assert offset == 5
        assert result == ItemSet([
            lex.Unit('a', Symbols.Ident),
            lex.Unit('b', Symbols.Ident)
        ])

    def test_empty(self):
        units = [
            lex.Unit('(', Symbols.LeftPar),
            lex.Unit(')', Symbols.RightPar)
        ]
        assert ItemList.parse(units) == (ItemList([]), 2)

    def test_nested(self):

        class Array(ItemList):
            left, right = Symbols.LeftSquareBracket, Symbols.RightSquareBracket

        units = [
            lex.Unit('(', Symbols.LeftPar),
                lex.Unit('[', Symbols.LeftSquareBracket),
                    lex.Unit('a', Symbols.Ident), lex.Unit(',', Symbols.Comma),
                    lex.Unit('b', Symbols.Ident),
                lex.Unit(']', Symbols.RightSquareBracket),
            lex.Unit(',', Symbols.Comma),
                lex.Unit('[', Symbols.LeftSquareBracket),
                    lex.Unit('c', Symbols.Ident), lex.Unit(',', Symbols.Comma),
                    lex.Unit('d', Symbols.Ident),
                lex.Unit(']', Symbols.RightSquareBracket),
            lex.Unit(')', Symbols.RightPar)
        ]

        assert ItemList[Array].parse(units) == (
            ItemList([
                Array([lex.Unit('a', Symbols.Ident), lex.Unit('b', Symbols.Ident)]),
                Array([lex.Unit('c', Symbols.Ident), lex.Unit('d', Symbols.Ident)])
            ]),
             len(units)
        )
    
    def test_Arg(self):

        units = [
            lex.Unit('(', Symbols.LeftPar),
            lex.Unit('a', Symbols.Ident),
            lex.Unit('str', Symbols.Ident),
            lex.Unit(')', Symbols.RightPar),
        ]

        assert ItemList[Arg].parse(units) == (
            ItemList([
                Arg(
                    name=lex.Unit('a', Symbols.Ident),
                    type=TypeExpr(value=lex.Unit('str', Symbols.Ident))
                )
            ]),
            4
        )

    # def test_no_trailing(self):
        
    #     units = [
    #         lex.Unit(None, Symbols.Ident),
    #         lex.Unit(None, Symbols.Comma)
    #     ]

    #     import pytest
    #     with pytest.raises(SyntaxError) as errinfo:
    #         ItemList.parse(units, no_trailing=True)