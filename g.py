from dataclasses import dataclass
from enum import Enum
from typing import Generic, List, Literal, Optional, Tuple, Any, Type, TypeVar, Union
import typing

import lex
import tree


T = TypeVar('T')


class Synt:
    value: Any

    def __init__(self, value: Any = None, **kw):
        self.value = value
        for key, value in kw.items():
            setattr(self, key, value)

    @classmethod
    def parse(cls, units: List[lex.Unit]) -> Tuple[Union['Synt', lex.Unit], int, Optional[Exception]]:
        if not units:
            raise ValueError('No units to parse!')
        return units[0], 1, None

    @classmethod
    def patterns(cls) -> List[List[Enum]]:
        return [[tree.Specials.Any]]
    
    @classmethod
    def tree(cls) -> tree.Node:
        start, end = tree.Node(), tree.Node(goal=cls)
        start >> tree.Specials.Any >> end
        return start, end


class Symbols(Enum):
    Ident = 'ident'
    LeftPar, RightPar = '(', ')'
    LeftBracket, RightBracket = '{', '}'
    LeftSquareBracket, RightSquareBracket = '[', ']'
    FuncKeyword = 'fn'
    Comma = ','
    NumberLiteral = r'\d+'


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
                obj, tskip, err = cls.T.parse(units)
                skip += tskip
                units = units[tskip:]
                items.append(obj)
            boat = not boat

        # expect a right par
        assert units[0].type == cls.right
        return getattr(cls, 'base', cls)(items), skip, None


class Chain(Synt, Generic[T]):

    T = Synt

    @classmethod
    def __class_getitem__(cls, target: type):
        return type(getattr(target, '__name__', getattr(target, 'name', None)) + 'Chain', (cls,), {'T': target})

    @classmethod
    def parse(cls, units):
        skip = 0
        objs = []
        while units:
            obj, tskip, err = cls.T.parse(units)
            objs.append(obj)
            skip += tskip
            units = units[tskip:]
        return objs, skip, None

    @classmethod
    def patterns(cls):
        pass


class ASynt(Synt):

    def __repr__(self):
        astr = ', '.join('%s=%s' % (key, getattr(self, key)) for key in self.__annotations__ if not key.startswith('_'))
        return '%s(%s)' % (self.__class__.__name__, astr)

    def __eq__(self, obj) -> bool:
        if not isinstance(obj, self.__class__):
            return False
        for key in self.__annotations__:
            if key.startswith('_'):
                continue
            if getattr(obj, key) != getattr(self, key):
                return False
        return True

    @classmethod
    def parse(cls, units):

        skip = 0
        data = {}

        for key, value in cls.__annotations__.items():

            required = True
            obj = None
            err = None
            tskip = 0

            while isinstance(value, typing._GenericAlias):
                required = False
                value, *_ = value.__args__

            if isinstance(value, Symbols):
                if units:
                    obj = units[0]
                    if obj.type != value:
                        obj = None
                    else:
                        tskip = 1
            else:
                obj, tskip, err = value.parse(units)

            if required and obj is None:
                return obj, tskip, err or Exception('Expected %s' % value)

            if not key.startswith('_'):
                data[key] = obj
            skip += tskip
            units = units[tskip:]

        return cls(**data), skip, None

    @classmethod
    def patterns(cls):
        pats = [[]]
        for _, value in cls.__annotations__.items():

            required = True
            while isinstance(value, typing._GenericAlias):
                required = False
                value, *_ = value.__args__
            
            if isinstance(value, Symbols):
                for p in list(pats):
                    if not required:
                        pats.append(list(p))
                    p.append(value)
            else:
                for p in value.patterns():
                    for i, q in enumerate(list(pats)):
                        pats[i] = [*q, *p]
                        if not required:
                            pats.append(list(q))
        return pats

    @classmethod
    def tree(cls):
        start = tree.Node()
        end   = tree.Node(goal=cls)
        for pat in cls.patterns():
            if not pat:
                continue
            start >> pat >> tree.Node(goal=cls)
        return start, end


class TestASynt:

    def test_basic(self):

        class Arg(ASynt):
            a: Symbols.Ident
            b: Symbols.Ident

        units = [
            lex.Unit('a', Symbols.Ident),
            lex.Unit('b', Symbols.Ident)
        ]

        result, offset, err = Arg.parse(units)
        assert result == Arg(
            a=lex.Unit('a', Symbols.Ident),
            b=lex.Unit('b', Symbols.Ident)
        )
        assert offset == 2
        assert err is None

    def test_patterns(self):

        class A(ASynt):
            name: Symbols.Ident
            type: Symbols.Ident

        assert A.patterns() == [[Symbols.Ident, Symbols.Ident]]

        class B(ASynt):
            a: A
            name: Symbols.Ident

        assert B.patterns() == [[Symbols.Ident, Symbols.Ident, Symbols.Ident]]

        class C(ASynt):
            start: Symbols.RightPar
            a: A
            b: B
            end: Symbols.LeftPar

        pats = C.patterns()
        print(pats)
        assert pats == [
            [
                Symbols.RightPar,
                    # a: A
                    Symbols.Ident, Symbols.Ident,
                    # b: B
                        # b: B -> a: A
                        Symbols.Ident, Symbols.Ident,
                        # b: B -> name
                        Symbols.Ident,
                Symbols.LeftPar
            ]
        ]

    def test_patterns_nested(self):

        class TypeExpr(ASynt):
            name: Symbols.Ident

        class Param(ASynt):
            name: Symbols.Ident
            type: TypeExpr

        class Func(ASynt):
            name: Symbols.Ident
            param: Param
            returnType: TypeExpr
        
        assert TypeExpr.patterns() == [[Symbols.Ident]]

        assert Param.patterns() == [
            [Symbols.Ident, Symbols.Ident],
        ]

        assert Func.patterns() == [
            [Symbols.Ident, Symbols.Ident, Symbols.Ident, Symbols.Ident]
        ]

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

        result, offset, err = Func.parse(units)
        assert offset == 6
        assert err is None
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

    def test_hidden(self):

        class A(ASynt):
            name: Symbols.Ident
            _c: Symbols.Comma
            value: Symbols.Ident
        
        units = [
            lex.Unit('a', Symbols.Ident),
            lex.Unit(',', Symbols.Comma),
            lex.Unit('b', Symbols.Ident),
        ]

        r, offset, err = A.parse(units)
        assert r == A(
            name=lex.Unit('a', Symbols.Ident),
            value=lex.Unit('b', Symbols.Ident)
        )
        assert offset == 3
        assert err is None
    
    def test_tree(self):

        class TypeExpr(ASynt):
            name: Symbols.Ident

        class Param(ASynt):
            name: Symbols.Ident
            type: TypeExpr

        class Func(ASynt):
            name: Symbols.Ident
            param: Param
            returnType: TypeExpr

        units = [Symbols.Ident, Symbols.Ident, Symbols.Ident, Symbols.Ident]

        root = tree.Node()
        leaf = tree.Node(goal=TypeExpr)
        root >> Symbols.Ident >> leaf
        t, end = TypeExpr.tree()
        assert root == t
        assert leaf == end
        assert t.last(units) == (TypeExpr, 1)

        root = tree.Node()
        leaf = tree.Node(goal=Param)
        root >> (Symbols.Ident, Symbols.Ident) >> leaf
        t, end = Param.tree()
        assert root == t
        assert leaf == end
        assert t.last(units) == (Param, 2)

        root = tree.Node()
        leaf = tree.Node(goal=Func)
        root >> (Symbols.Ident, Symbols.Ident, Symbols.Ident, Symbols.Ident) >> leaf
        t, end = Func.tree()
        assert root == t
        assert leaf == end
        assert t.last(units) == (Func, 4)

    class TestOptional:

        class Arg(ASynt):
            a: Optional[Literal[Symbols.Ident]]
            b: Optional[Literal[Symbols.NumberLiteral]]

        def test_seamless(self):

            units = [
                lex.Unit('a', Symbols.Ident),
                lex.Unit('b', Symbols.NumberLiteral)
            ]

            result, offset, err = self.Arg.parse(units)
            assert result == self.Arg(
                    a=lex.Unit('a', Symbols.Ident),
                    b=lex.Unit('b', Symbols.NumberLiteral)
                )
            assert offset == 2
            assert err is None

        def test_basic(self):

            units = [
                lex.Unit('a', Symbols.Ident),
            ]

            result, offset, err = self.Arg.parse(units)
            assert result == self.Arg(
                    a=lex.Unit('a', Symbols.Ident),
                    b=None
                )
            assert offset == 1
            assert err is None

        def test_skip(self):

            units = [
                lex.Unit('a', Symbols.NumberLiteral),
            ]

            result, offset, err = self.Arg.parse(units)
            assert result == self.Arg(
                    a=None,
                    b=lex.Unit('a', Symbols.NumberLiteral)
                )
            assert offset == 1
            assert err is None

        def test_early_optional(self):

            class Arg(ASynt):
                a: Optional[Literal[Symbols.NumberLiteral]]
                b: Symbols.Ident

            units = [
                lex.Unit('a', Symbols.NumberLiteral),
            ]

            result, offset, err = Arg.parse(units)
            assert result is None
            assert offset == 0
            assert err is not None
            assert str(err) == 'Expected %s' % Symbols.Ident

        def test_nested(self):
            
            class A(ASynt):
                name: Symbols.Ident
            
            class B(ASynt):
                _left: Symbols.LeftPar
                a: Optional[A]
                _right: Symbols.RightPar

            units = [
                lex.Unit(None, Symbols.LeftPar),
                lex.Unit(None, Symbols.RightPar)
            ]

            result, offset, err = B.parse(units)
            assert result == B(a=None)
            assert offset == 2
            assert err is None

            units = [
                lex.Unit(None, Symbols.LeftPar),
                lex.Unit('a', Symbols.Ident),
                lex.Unit(None, Symbols.RightPar)
            ]

            result, offset, err = B.parse(units)
            assert result == B(a=A(name=lex.Unit('a', Symbols.Ident)))
            assert offset == 3
            assert err is None
        
        def test_patterns(self):

            assert self.Arg.patterns() == [
                [Symbols.Ident, Symbols.NumberLiteral],
                [Symbols.NumberLiteral],
                [Symbols.Ident],
                []
            ]

        def test_patterns_nested(self):

            class A(ASynt):
                name: Symbols.Ident
            
            class B(ASynt):
                _left: Symbols.LeftPar
                a: Optional[A]
                _right: Symbols.RightPar

            assert A.patterns() == [
                [Symbols.Ident]
            ]
            
            assert B.patterns() == [
                [Symbols.LeftPar, Symbols.Ident, Symbols.RightPar],
                [Symbols.LeftPar, Symbols.RightPar],
            ]
        
        def test_tree(self):
            a, b, c, d = tree.Node.make(4)
            b.goal = c.goal = d.goal = self.Arg

            a.children = {
                Symbols.Ident: b,
                Symbols.NumberLiteral: d
            }

            b.children = {
                Symbols.NumberLiteral: c
            }

            t, end = self.Arg.tree()
            a.print()
            t.print()
            assert a.assert_equals(t)
            assert t.last([]) == (False, 0)  # you can't have empty patterns anyway
            assert t.last([Symbols.Ident]) == (self.Arg, 1)
            assert t.last([Symbols.NumberLiteral]) == (self.Arg, 1)
            assert t.last([Symbols.Ident, Symbols.NumberLiteral]) == (self.Arg, 2)
            assert t.last([Symbols.NumberLiteral, Symbols.Ident]) == (self.Arg, 1)

class TestItemList:

    def test_basic(self):

        units = [
            lex.Unit('(', Symbols.LeftPar),
            lex.Unit('a', Symbols.Ident),
            lex.Unit(',', Symbols.Comma),
            lex.Unit('b', Symbols.Ident),
            lex.Unit(')', Symbols.RightPar)
        ]

        result, offset, err = ItemList.parse(units)
        assert offset == 5
        assert err is None
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

        result, offset, err = ItemSet.parse(units)
        assert offset == 5
        assert err is None
        assert result == ItemSet([
            lex.Unit('a', Symbols.Ident),
            lex.Unit('b', Symbols.Ident)
        ])

    def test_empty(self):
        units = [
            lex.Unit('(', Symbols.LeftPar),
            lex.Unit(')', Symbols.RightPar)
        ]
        result, offset, err = ItemList.parse(units)
        assert result == ItemList([])
        assert offset == 2
        assert err is None

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

        result, offset, err = ItemList[Array].parse(units)
        assert  result == ItemList([
            Array([lex.Unit('a', Symbols.Ident), lex.Unit('b', Symbols.Ident)]),
            Array([lex.Unit('c', Symbols.Ident), lex.Unit('d', Symbols.Ident)])
        ])
        assert offset == len(units)
        assert err is None

    # def test_patterns(self):
        
    #     start = Node()
    #     item  = Node()
    #     end   = Node()

    #     start >> Symbols.LeftPar >> item
    #     item  >> Any >> Symbols.Comma >> item
    #     item  >> Any >> end
    #     start >> end

    #     assert ItemList.tree() == n

    # def test_patterns_nested(self):
    #     assert False


class TestChain:

    def test_basic(self):

        class TypeExpr(ASynt):
            name: Symbols.Ident

        units = [
            lex.Unit('a', Symbols.Ident),
            lex.Unit('b', Symbols.Ident),
            lex.Unit('a', Symbols.Ident),
        ]

        result, offset, err = Chain[TypeExpr].parse(units)
        assert offset == 3
        assert err is None
        assert result == [
            TypeExpr(name=lex.Unit('a', Symbols.Ident)),
            TypeExpr(name=lex.Unit('b', Symbols.Ident)),
            TypeExpr(name=lex.Unit('a', Symbols.Ident))
        ]


@dataclass
class SyntUnion(Synt):
    candidates: List[Type[Synt]]

    @classmethod
    def build(cls):
        from trie import Trie
        if hasattr(cls, 'tries'):
            return
        cls.tries = {
            c: Trie().init(*c.patterns())
            for c in cls.candidates
        }

    @classmethod
    def parse(cls, units):

        # build a trie out of each candidate's pattern
        cls.build()

        # take the longest result

        result = None
        maxsize = 0

        for c, trie in cls.tries.items():
            found, size = trie.find(units)
            if found and size > maxsize:
                maxsize = size
                result = c

        return result


class TestUnion:

    def test_basic(self):

        class Var(ASynt):
            name: Symbols.Ident
            type: Symbols.Ident

        assert Var.patterns() == [
            [Symbols.Ident, Symbols.Ident]
        ]

        class IdentOrVar(SyntUnion):
            candidates = [Var, Symbols.Ident]


__all__ = ['Synt', 'ASynt', 'Chain', 'ItemList', 'Symbols']
