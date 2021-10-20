from dataclasses import dataclass
from enum import Enum
from typing import Dict, Generic, List, Literal, Optional, Tuple, Any, Type, TypeVar, Union
import typing

import lex
import tree


T = TypeVar('T')


class Synt:
    value: Any

    _templated: Dict[Any, Type['Synt']] = {}

    @classmethod
    def __class_getitem__(cls, v):
        n = cls._templated.get(v)
        if n is not None:
            return n
        class n(cls):
            value = v
        cls._templated[v] = n
        return n

    def __init__(self, value: Any = None, **kw):
        self.value = value
        kw = {
            **{k: None for k in self.__annotations__},
            **kw
        }
        for key, value in kw.items():
            setattr(self, key, value)

    @classmethod
    def parse(cls, units: List[lex.Unit]) -> Tuple[Union['Synt', lex.Unit], int, Optional[Exception]]:
        if not units:
            raise ValueError('No units to parse!')
        return units[0], 1, None

    @classmethod
    def patterns(cls) -> List[List[Enum]]:
        return [[cls.value or tree.Specials.Any]]
    
    @classmethod
    def tree(cls) -> tree.Node:
        start, end = tree.Node(), tree.Node(goal=cls)
        start >> (cls.value or tree.Specials.Any) >> end
        return start, [end]


class Symbols(Enum):
    Ident = 'ident'
    LeftPar, RightPar = '(', ')'
    LeftBracket, RightBracket = '{', '}'
    LeftSquareBracket, RightSquareBracket = '[', ']'
    FuncKeyword = 'fn'
    Comma = ','
    NumberLiteral = r'\d+'
    Slash = '/'


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
        raise NotImplementedError
    
    @classmethod
    def tree(cls):
        t, ends = cls.T.tree()
        for e in ends:
            e.goal = cls
            e.children[tree.Specials.Empty] = t
        return t, ends


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
                value_pats = value.patterns()
                if not value_pats:
                    continue
                for i, q in enumerate(list(pats)):
                    first = value_pats[0]
                    pats[i] = [*q, *first]
                    for p in value_pats[1:]:
                        pats.append([*q, *p])
                    if not required:
                        pats.append(list(q))
                
        return pats

    @classmethod
    def tree(cls):
        start = tree.Node()
        ends = []
        for pat in cls.patterns():
            if not pat:
                continue
            ends.append(start >> pat >> tree.Node(goal=cls))
        return start, ends


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
        t, ends = TypeExpr.tree()
        assert root == t
        assert t.last(units) == (TypeExpr, 1)

        root = tree.Node()
        leaf = tree.Node(goal=Param)
        root >> (Symbols.Ident, Symbols.Ident) >> leaf
        t, ends = Param.tree()
        assert root == t
        assert t.last(units) == (Param, 2)

        root = tree.Node()
        leaf = tree.Node(goal=Func)
        root >> (Symbols.Ident, Symbols.Ident, Symbols.Ident, Symbols.Ident) >> leaf
        t, ends = Func.tree()
        assert root == t
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

            t, ends = self.Arg.tree()
            a.print()
            t.print()
            assert a.assert_equals(t)
            assert t.last([]) == (False, 0)  # you can't have empty patterns anyway
            assert t.last([Symbols.Ident]) == (self.Arg, 1)
            assert t.last([Symbols.NumberLiteral]) == (self.Arg, 1)
            assert t.last([Symbols.Ident, Symbols.NumberLiteral]) == (self.Arg, 2)
            assert t.last([Symbols.NumberLiteral, Symbols.Ident]) == (self.Arg, 1)

        def test_tree_nested(self):
        
            class A(ASynt):
                name: Symbols.Ident
            
            class B(ASynt):
                _left: Symbols.LeftPar
                a: Optional[A]
                _right: Symbols.RightPar

            a, b, c, d, e = tree.Node.make(5)
            d.goal = e.goal = B

            a.children = { Symbols.LeftPar: b }
            b.children = {
                Symbols.Ident: c,
                Symbols.RightPar: e
            }
            c.children = { Symbols.RightPar: d }

            t, ends = B.tree()
            a.print()
            t.print()
            assert a.assert_equals(t)
            assert t.last([Symbols.LeftPar, Symbols.RightPar]) == (B, 2)
            assert t.last([Symbols.LeftPar, Symbols.Ident, Symbols.RightPar]) == (B, 3)
            assert t.last([Symbols.LeftPar, Symbols.Ident, Symbols.NumberLiteral]) == (False, 3)
            assert t.last([Symbols.LeftPar, Symbols.NumberLiteral, Symbols.RightPar]) == (False, 2)


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
    
    def test_patterns(self):
        import pytest
        with pytest.raises(NotImplementedError):
            Chain.patterns()

    def test_tree(self):

        class TypeExpr(ASynt):
            name: Symbols.Ident
        
        c = Chain[TypeExpr]
        t, ends = c.tree()

        a, b = tree.Node(),tree.Node(goal=TypeExpr)
        a.children = {
            Symbols.Ident: b
        }

        tet, _ = TypeExpr.tree()
        assert tet.assert_equals(a)

        a.children[Symbols.Ident].children = { tree.Specials.Empty: a }
        a.children[Symbols.Ident].goal = c
        # assert t.assert_equals(a)  # FIXME recursive error

        assert t.last([Symbols.Ident]) == (c, 1)
        assert t.last([Symbols.Ident, Symbols.Ident]) == (c, 2)
        assert t.last([Symbols.Ident, Symbols.Ident, Symbols.NumberLiteral]) == (c, 2)
    
    def test_tree_nested(self):

        class Rhythm(ASynt):
            _left: Symbols.LeftPar
            value: Symbols.NumberLiteral

        class Note(ASynt):
            name: Symbols.Ident
            octave: Optional[Literal[Symbols.NumberLiteral]]
            alteration: Optional[Literal[Symbols.Ident]]
            rhythm: Optional[Rhythm]
        
        t, ends = Note.tree()

        a, b, c, d, e, f, g, h, i, j, k, l, m = tree.Node.make(13)
        for n in (b,c, d, f, h, i, k, m):
            n.goal = Note

        a.children = { Symbols.Ident: b }
        g.children = { Symbols.NumberLiteral: h }
        e.children = { Symbols.NumberLiteral: f }
        l.children = { Symbols.NumberLiteral: m }
        j.children = { Symbols.NumberLiteral: k }
        b.children = {
            Symbols.NumberLiteral: c,
            Symbols.Ident: d,
            Symbols.LeftPar: e,
        }
        c.children = {
            Symbols.Ident: i,
            Symbols.LeftPar: l,
        }
        d.children = { Symbols.LeftPar: g }
        i.children = { Symbols.LeftPar: j }

        # # Ideal version
        # a, b, c, d, e, f = tree.Note.make(6)
        # a.children = { Symbols.Ident: b }
        # b.children = {
        #     Symbols.NumberLiteral: c,
        #     Symbols.Ident: d,
        #     Symbols.LeftPar: e
        # }
        # c.children = {
        #     Symbols.Ident: d,
        #     Symbols.LeftPar: e
        # }
        # d.children = { Symbols.LeftPar: e }
        # e.children = { Symbols.NumberLiteral: f }
        
        a.print()
        t.print()

        assert t.assert_equals(a)

        assert t.last([Symbols.Ident]) == (Note, 1)
        assert t.last([Symbols.Ident, Symbols.Ident]) == (Note, 2)
        assert t.last([Symbols.Ident, Symbols.NumberLiteral]) == (Note, 2)
        assert t.last([Symbols.Ident, Symbols.NumberLiteral, Symbols.Ident]) == (Note, 3)
        assert t.last([Symbols.Ident, Symbols.NumberLiteral, Symbols.Ident, Symbols.LeftPar]) == (Note, 3)
        assert t.last([Symbols.Ident, Symbols.NumberLiteral, Symbols.Ident, Symbols.LeftPar, Symbols.NumberLiteral]) == (Note, 5)


@dataclass
class SyntUnion(Synt):
    candidates: List[Type[Synt]]

    @classmethod
    def __class_getitem__(cls, c):
        if not isinstance(c, tuple):
            c = (c,)
        c = list(c)
        for i, item in enumerate(c):
            if isinstance(item, Enum):
                item = Synt[item]
            c[i] = item
        class n(cls):
            candidates = c
        n.__name__ = 'SyntUnion'
        return n

    @classmethod
    def patterns(cls):
        cls.build()
        def f():
            for c in cls.candidates:
                yield from c.patterns() 
        return list(f())

    @classmethod
    def tree(cls):
        start = tree.Node()
        ends = []
        cls.build()
        for c in cls.candidates:
            s, e = c.tree()
            start |= s
            ends.extend(e)
        return start, ends

    @classmethod
    def build(cls):
        if hasattr(cls, 'tries'):
            return
        cls.tries = {
            c: c.tree()[0]
            for c in cls.candidates
        }

    @classmethod
    def detect(cls, units):

        # build a trie out of each candidate's pattern
        cls.build()

        # take the longest result
        result = None
        maxsize = 0

        for c, trie in cls.tries.items():
            found, size = trie.last(units)
            if found and size > maxsize:
                maxsize = size
                result = c

        return result

    @classmethod
    def parse(cls, units):
        p = cls.detect([u.type for u in units])
        if p is None:
            return 0, None
        return p.parse(units)


class TestUnion:

    def test_basic(self):

        class Var(ASynt):
            name: Symbols.Ident
            type: Symbols.Ident

        assert Var.patterns() == [
            [Symbols.Ident, Symbols.Ident]
        ]

        s = Synt[Symbols.Ident]
        IdentOrVar = SyntUnion[Var, s]
        
        assert IdentOrVar.detect([Symbols.Ident]) == s
        assert IdentOrVar.detect([Symbols.Ident, Symbols.Ident]) == Var

    def test_nested(self):

        class Rhythm(ASynt):
            _left: Symbols.Slash
            value: Symbols.NumberLiteral

        class Note(ASynt):
            name: Symbols.Ident
            octave: Optional[Literal[Symbols.NumberLiteral]]
            alteration: Optional[Literal[Symbols.Ident]]
            rhythm: Optional[Rhythm]

        U = SyntUnion[Rhythm, Note]
                
        units = [
            lex.Unit('a', Symbols.Ident),
            lex.Unit('/', Symbols.Slash),
            lex.Unit('33', Symbols.NumberLiteral),
            lex.Unit('a', Symbols.Ident)
        ]

        assert U.detect([u.type for u in units]) == Note
        r, offset, err = U.parse(units)
        assert offset == 3
        assert err is None
        assert r == Note(
            name=lex.Unit('a', Symbols.Ident),
            rhythm=Rhythm(
                value=lex.Unit('33', Symbols.NumberLiteral)
            )
        )

        units = units[1:]
        assert U.detect([u.type for u in units]) == Rhythm
        r, offset, err = U.parse(units)
        assert offset == 2
        assert err is None
        assert r == Rhythm(value=lex.Unit('33', Symbols.NumberLiteral))

    def test_patterns(self):

        U = SyntUnion[Symbols.Ident, Symbols.NumberLiteral]
        assert U.patterns() == [
            [Symbols.Ident],
            [Symbols.NumberLiteral]
        ]

    def test_tree(self):

        U = SyntUnion[Symbols.Ident, Symbols.NumberLiteral]
        t, ends = U.tree()

        a, b, c = tree.Node.make(3)
        b.goal = Synt[Symbols.Ident]
        c.goal = Synt[Symbols.NumberLiteral]
        a.children = { Symbols.Ident: b, Symbols.NumberLiteral: c }

        t.print()
        a.print()

        assert t.assert_equals(a)
        assert all(a.assert_equals(b) for a, b in zip(ends, [b, c]))

        assert t.last([Symbols.Ident]) == (Synt[Symbols.Ident], 1)
        assert t.last([Symbols.Ident, Symbols.NumberLiteral]) == (Synt[Symbols.Ident], 1)
        assert t.last([Symbols.NumberLiteral]) == (Synt[Symbols.NumberLiteral], 1)

    def test_tree_nested(self):
        
        class B(ASynt):
            value: Symbols.Ident
            type: Symbols.Ident

        class A(ASynt):
            value: SyntUnion[B, Symbols.NumberLiteral]
        
        class X(ASynt):
            a: A
            b: Symbols.NumberLiteral
        
        t, ends = X.tree()

        a, b, c, d, e, f = tree.Node.make(6)
        e.goal = f.goal = X

        a.children = { Symbols.Ident: b, Symbols.NumberLiteral: c }
        b.children = { Symbols.Ident: d }
        c.children = { Symbols.NumberLiteral: f }
        d.children = { Symbols.NumberLiteral: e }

        assert a.assert_equals(t)
        assert all(a.assert_equals(b) for a, b in zip(ends, [e, f]))

        assert t.last([Symbols.Ident, Symbols.Ident, Symbols.NumberLiteral]) == (X, 3)
        assert t.last([Symbols.NumberLiteral, Symbols.NumberLiteral]) == (X, 2)
        assert t.last([Symbols.Ident, Symbols.NumberLiteral, Symbols.NumberLiteral]) == (False, 2)

        r, offset, err = X.parse([lex.Unit("a", Symbols.Ident), lex.Unit("b", Symbols.Ident), lex.Unit(None, Symbols.NumberLiteral)])
        assert offset == 3
        assert err is None
        assert r == X(
            a=A(
                value=B(
                    value=lex.Unit("a", Symbols.Ident),
                    type=lex.Unit("b", Symbols.Ident),
                )
            ),
            b=lex.Unit(None, Symbols.NumberLiteral)
        )

        r, offset, err = X.parse([lex.Unit('33', Symbols.NumberLiteral), lex.Unit('333', Symbols.NumberLiteral)])
        assert offset == 2
        assert err is None
        assert r == X(
            a=A(
                value=lex.Unit('33', Symbols.NumberLiteral)
            ),
            b=lex.Unit('333', Symbols.NumberLiteral)
        )

    def test_chain(self):

        Instruction = SyntUnion[Symbols.Ident, Symbols.NumberLiteral]
        Block = Chain[Instruction]

        # first, tree
        t, ends = Block.tree()

        a, b, c = tree.Node.make(3)
        b.goal = c.goal = Block
        a.children = { Symbols.Ident: b, Symbols.NumberLiteral: c }
        b.children = c.children = { tree.Specials.Empty: a }

        assert a.assert_equals(t)

        units = [lex.Unit('a', Symbols.Ident), lex.Unit('b', Symbols.Ident), lex.Unit('11', Symbols.NumberLiteral)]
        assert t.last([u.type for u in units]) == (Block, 3)

        r, offset, err = Block.parse(units)
        assert offset == 3
        assert err is None
        assert r == units   


__all__ = ['Synt', 'ASynt', 'Chain', 'ItemList', 'Symbols']
