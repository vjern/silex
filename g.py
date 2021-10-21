from dataclasses import dataclass
from enum import Enum
import traceback
from sys import meta_path
from typing import Dict, ForwardRef, Generic, List, Literal, Optional, Tuple, Any, Type, TypeVar, Union
import typing

import lex
from tree import Node, Specials


T = TypeVar('T')
__all__ = ['Synt', 'ASynt', 'Chain', 'ItemList', 'Symbols']


def resolve_raw_forward_ref(name: str, modulename: str, ctx: dict = None) -> type:
    import sys
    module = sys.modules[modulename]
    print(f'{modulename=} {module}')
    result = None
    if ctx is not None:
        result = ctx.get(name)
    return result or getattr(module, name, None)


class ForwardRefResolver:
    @classmethod
    def resolve_forward_refs(cls, ctx=None):
        raise NotImplementedError


class SyntGeneric(ForwardRefResolver, Generic[T]):

    _templated = {}
    
    @classmethod
    def __class_template__(cls, t) -> Tuple[str, Tuple[type, ...], dict]:
        return '%s<%s>' % (cls.__name__, getattr(t, '__name__', str(t))), (cls,), {'T': t}

    @classmethod
    def __class_getitem__(cls, t):
        n = cls._templated.get((cls, t))
        if n is not None:
            return n
        n = cls._templated[cls, t] = type(*cls.__class_template__(t))
        return n
    
    @classmethod
    def resolve_forward_refs(cls, ctx=None):
        value = cls.T
        optional = False
        while isinstance(value, typing._GenericAlias):
            if value.__origin__ is Union:
                optional = True
            value, *_ = value.__args__
        if isinstance(value, str):
            value = resolve_raw_forward_ref(value, cls.__module__, ctx)
        elif isinstance(value, ForwardRef):
            value = value._evaluate(globals(), ctx)
        elif isinstance(value, type) and issubclass(value, ForwardRefResolver):
            value.resolve_forward_refs(ctx)
        if optional:
            value = Optional[value]
        cls.T = value


class TestSyntGeneric:

    def test_basic(self):
        T = int
        s = SyntGeneric[T]
        d = SyntGeneric[T]
        assert s.__name__ == 'SyntGeneric<int>'
        assert s is not SyntGeneric
        assert s is d
        assert s.T is T


class Synt(SyntGeneric):

    T = None

    def __init__(self, **kw):
        kw = {
            **{k: None for k in self.__annotations__},
            **kw
        }
        for key, value in kw.items():
            setattr(self, key, value)

    @classmethod
    def parse(cls, units: List[lex.Unit]) -> Tuple[Union['Synt', lex.Unit], int, Optional[Exception]]:
        if not units:
            return None, 0, ValueError('No units to parse!')
        unit = units[0]
        if cls.T is not None and unit.type != cls.T:
            return None, 0, Exception('Expected %s' % cls.T)
        return unit, 1, None

    @classmethod
    def patterns(cls) -> List[List[Enum]]:
        return [[cls.T or Specials.Any]]
    
    @classmethod
    def tree(cls) -> Tuple[Node, List[Node]]:
        start, end = Node(), Node(goal=cls)
        start >> (cls.T or Specials.Any) >> end
        return start, [end]
    
    @classmethod
    def xtree(cls) -> Tuple[Node, List[Node]]:
        return cls.tree()


class Symbols(Enum):
    Ident = 'ident'
    LeftPar, RightPar = '(', ')'
    LeftBracket, RightBracket = '{', '}'
    LeftSquareBracket, RightSquareBracket = '[', ']'
    FuncKeyword = 'fn'
    Comma = ','
    NumberLiteral = r'\d+'
    Slash = '/'
    Dash = '-'
    Underscore = '_'
    Plus = '+'
    Equals = '='


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


class Chain(Synt, SyntGeneric):

    T = Synt

    @classmethod
    def parse(cls, units):
        skip = 0
        objs = []
        while units:
            obj, tskip, err = cls.T.parse(units)
            if err is not None:
                break
            print('success chain', obj)
            objs.append(obj)
            skip += tskip
            units = units[tskip:]
        print('Chain.parse returns', objs, skip)
        return objs, skip, None

    @classmethod
    def patterns(cls):
        raise NotImplementedError
    
    @classmethod
    def tree(cls):
        t, ends = cls.T.xtree()
        for e in ends:
            e.goal = cls
            e.children[Specials.Empty] = t
        return t, ends
        for e in ends:
            t = t.merge(e, inplace=True)
            # FIXME this cuts the relationship between e and its parent which we want to keep
        t.goal = cls
        return t, [t]


class ASynt(Synt, ForwardRefResolver):

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
    def resolve_forward_refs(cls, ctx=None):
        ants = cls.__annotations__
        for key, value in ants.items():
            optional = False
            while isinstance(value, typing._GenericAlias):
                if value.__origin__ is Union:
                    optional = True
                value, *_ = value.__args__
            if isinstance(value, str):
                value = resolve_raw_forward_ref(value, cls.__module__, ctx)
            elif isinstance(value, ForwardRef):
                value = value._evaluate(globals(), ctx)
            elif isinstance(value, type) and issubclass(value, ForwardRefResolver):
                value.resolve_forward_refs(ctx)
            if optional:
                value = Optional[value]
            ants[key] = value

    @classmethod
    def parse(cls, units):

        print(f'{cls.__name__}.parse')

        skip = 0
        data = {}

        for key, value in cls.__annotations__.items():

            print(f'{units=}')
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
                        # if required:
                        #     raise Exception('ga %s %s' % (obj.type, value))
                        obj = None
                    else:
                        tskip = 1
            else:
                obj, tskip, err = value.parse(units)
                print('our warriors have returned!', obj, tskip, err, value)

            if required and (obj is None or err):
                return obj, tskip, err or Exception('%s:%s:Expected %s, found %s' % (cls.__name__, key, value, obj))

            if not key.startswith('_'):
                data[key] = obj

            skip += tskip
            units = units[tskip:]

        print('about to', cls, data)
        return cls(**data), skip, None

    @classmethod
    def patterns(cls):
        pats = [[]]
        for _, value in cls.__annotations__.items():

            required = True
            while isinstance(value, typing._GenericAlias):
                required = False
                value, *_ = value.__args__
            
            if isinstance(value, str):
                value = resolve_raw_forward_ref(value, cls.__module__)

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
        start = Node()
        ends = []
        for pat in cls.patterns():
            if not pat:
                start.goal = cls
                ends.append(start)
                continue
            ends.append(start >> pat >> Node(goal=cls))
        return start, ends

    @classmethod
    def xtree(cls):
        
        start = Node(name='start')
        nodes = [start]
        
        for _, value in cls.__annotations__.items():

            required = True
            while isinstance(value, typing._GenericAlias):
                required = False
                value, *_ = value.__args__
            
            if isinstance(value, str):
                value = resolve_raw_forward_ref(value, cls.__module__)

            if isinstance(value, Symbols):
                for i, node in enumerate(list(nodes)):
                    if not required:
                        nodes.append(node)
                    nodes[i] = node >> value >> Node()
            else:
                t, ends = value.xtree()
                for e in ends:
                    e.goal = None
                for i, node in enumerate(list(nodes)):
                    node.merge(t, inplace=True)
                    if not required:
                        ends.append(node)
                nodes = ends

        for n in nodes:
            n.goal = cls

        return start, nodes


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
    
    def test_forward_refs(self):

        class Assignment(ASynt):
            name: Symbols.Ident
            _eq: Symbols.LeftPar
            value: 'Expr'
            value2: ForwardRef('Expr')

        class Expr(ASynt):
            value: Symbols.NumberLiteral

        Assignment.resolve_forward_refs(locals())
        assert Assignment.__annotations__['value'] is Expr
        assert Assignment.__annotations__['value2'] is Expr

        assert Assignment.patterns() == [
            [Symbols.Ident, Symbols.LeftPar, Symbols.NumberLiteral, Symbols.NumberLiteral]
        ]

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

        root = Node()
        leaf = Node(goal=TypeExpr)
        root >> Symbols.Ident >> leaf
        t, ends = TypeExpr.tree()
        assert root.assert_equals(t)
        assert len(ends) == 1 and all(a.assert_equals(b) for a, b in zip(ends, [leaf]))
        assert t.last(units) == (TypeExpr, 1)

        root = Node()
        leaf = Node(goal=Param)
        root >> (Symbols.Ident, Symbols.Ident) >> leaf
        t, ends = Param.tree()
        assert root.assert_equals(t)
        assert len(ends) == 1 and all(a.assert_equals(b) for a, b in zip(ends, [leaf]))
        assert t.last(units) == (Param, 2)

        root = Node()
        leaf = Node(goal=Func)
        root >> (Symbols.Ident, Symbols.Ident, Symbols.Ident, Symbols.Ident) >> leaf
        t, ends = Func.tree()
        assert root.assert_equals(t)
        assert len(ends) == 1 and all(a.assert_equals(b) for a, b in zip(ends, [leaf]))
        assert t.last(units) == (Func, 4)

    def test_xtree(self):

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

        root = Node()
        leaf = Node(goal=TypeExpr)
        root >> Symbols.Ident >> leaf
        t, ends = TypeExpr.xtree()
        print('t =')
        t.print()
        assert root.assert_equals(t)
        assert leaf.assert_equals(ends[0])
        assert t.last(units) == (TypeExpr, 1)

        root = Node()
        leaf = Node(goal=Param)
        root >> (Symbols.Ident, Symbols.Ident) >> leaf
        t, ends = Param.xtree()
        print('t =')
        t.print()
        assert root.assert_equals(t)
        assert leaf.assert_equals(ends[0])
        assert t.last(units) == (Param, 2)

        root = Node()
        leaf = Node(goal=Func)
        root >> (Symbols.Ident, Symbols.Ident, Symbols.Ident, Symbols.Ident) >> leaf
        t, ends = Func.xtree()
        print('t =')
        t.print()
        assert root.assert_equals(t)
        assert leaf.assert_equals(ends[0])
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

            a, b, c, d = Node.make(4)
            a.goal = b.goal = c.goal = d.goal = self.Arg

            a.children = {
                Symbols.Ident: b,
                Symbols.NumberLiteral: d
            }
            b.children = { Symbols.NumberLiteral: c }

            t, ends = self.Arg.tree()
            a.print()
            t.print()
            assert a.assert_equals(t)  # FIXME always have expected node on the same side of assert_equals for consistency's sake
            assert all(b.print() or a.assert_equals(b) for a, b in zip(ends, [b, c, d][::-1]))

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

            a, b, c, d, e = Node.make(5)
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
            assert all(a.assert_equals(b) for a, b in zip(ends, [d, e]))

            assert t.last([Symbols.LeftPar, Symbols.RightPar]) == (B, 2)
            assert t.last([Symbols.LeftPar, Symbols.Ident, Symbols.RightPar]) == (B, 3)
            assert t.last([Symbols.LeftPar, Symbols.Ident, Symbols.NumberLiteral]) == (False, 3)
            assert t.last([Symbols.LeftPar, Symbols.NumberLiteral, Symbols.RightPar]) == (False, 2)
        
        def test_forward_refs(self):

            class Assignment(ASynt):
                name: Symbols.Ident
                _eq: Symbols.LeftPar
                value: Optional['Expr']
                value2: Optional[ForwardRef('Expr')]

            class Expr(ASynt):
                value: Symbols.NumberLiteral

            Assignment.resolve_forward_refs(locals())
            assert Assignment.__annotations__['value'] is Optional[Expr]
            assert Assignment.__annotations__['value2'] is Optional[Expr]

            pats = Assignment.patterns()
            print(len(pats), *pats, sep='\n')
            assert pats == [
                [Symbols.Ident, Symbols.LeftPar, Symbols.NumberLiteral, Symbols.NumberLiteral],
                [Symbols.Ident, Symbols.LeftPar, Symbols.NumberLiteral],
                [Symbols.Ident, Symbols.LeftPar, Symbols.NumberLiteral],  # odd but makes sense
                [Symbols.Ident, Symbols.LeftPar],
            ]

        def test_xtree(self):

            a, b, c, d = Node.make(4)
            a.goal = b.goal = c.goal = d.goal = self.Arg

            a.children = {
                Symbols.Ident: b,
                Symbols.NumberLiteral: d
            }
            b.children = { Symbols.NumberLiteral: c }

            t, ends = self.Arg.xtree()
            a.print()
            t.print()
            assert a.assert_equals(t)
            assert all(b.print() or a.assert_equals(b) for a, b in zip(ends, [b, c, d][::-1]))

            assert t.last([]) == (False, 0)  # you can't have empty patterns anyway
            assert t.last([Symbols.Ident]) == (self.Arg, 1)
            assert t.last([Symbols.NumberLiteral]) == (self.Arg, 1)
            assert t.last([Symbols.Ident, Symbols.NumberLiteral]) == (self.Arg, 2)
            assert t.last([Symbols.NumberLiteral, Symbols.Ident]) == (self.Arg, 1)

        def test_xtree_nested(self):
            
            class A(ASynt):
                name: Symbols.Ident
            
            class B(ASynt):
                _left: Symbols.LeftPar
                a: Optional[A]
                _right: Symbols.RightPar

            a, b, c, d, e = Node.make(5)
            d.goal = e.goal = B

            a.children = { Symbols.LeftPar: b }
            b.children = {
                Symbols.Ident: c,
                Symbols.RightPar: e
            }
            c.children = { Symbols.RightPar: d }

            t, ends = B.xtree()
            a.print()
            t.print()
            assert a.assert_equals(t)
            assert all(a.assert_equals(b) for a, b in zip(ends, [d, e]))

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

        a, b = Node(),Node(goal=TypeExpr)
        a.children = {
            Symbols.Ident: b
        }

        tet, _ = TypeExpr.tree()
        assert tet.assert_equals(a)

        a.children[Symbols.Ident].children = { Specials.Empty: a }
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

        a, b, c, d, e, f, g, h, i, j, k, l, m = Node.make(13)
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
        start = Node()
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
            c: c.xtree()[0]
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
        # p = cls.detect([u.type for u in units])
        for c in cls.candidates:
            obj, skip, err = c.parse(units)
            if not err:
                print(cls, 'found candidate', c, obj,)
                return (obj, skip, err)
        return None, 0, Exception('Found no candidate')


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

        a, b, c = Node.make(3)
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

        a, b, c, d, e, f = Node.make(6)
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

        a, b, c = Node.make(3)
        b.goal = c.goal = Block
        a.children = { Symbols.Ident: b, Symbols.NumberLiteral: c }
        b.children = c.children = { Specials.Empty: a }

        assert a.assert_equals(t)
        assert all(a.assert_equals(b) for a, b in zip(ends, [b, c]))

        units = [lex.Unit('a', Symbols.Ident), lex.Unit('b', Symbols.Ident), lex.Unit('11', Symbols.NumberLiteral)]
        assert t.last([u.type for u in units]) == (Block, 3)

        r, offset, err = Block.parse(units)
        assert offset == 3
        assert err is None
        assert r == units   


class TestHard:

    def test_basic(self):

        class Block(ASynt):
            _left: Symbols.LeftBracket
            instructions: Chain['Instruction']
            _right: Symbols.RightBracket
        
        class Instruction(ASynt):
            value: SyntUnion[Block, Symbols.Ident]

        Block.resolve_forward_refs(locals())
        assert Block.__annotations__['instructions'].T is Instruction

        units = [
            lex.Unit('{', Symbols.LeftBracket),
            lex.Unit('}', Symbols.RightBracket)
        ]

        r, offset, err = Block.parse(units)
        assert err is None
        assert offset == 2
        assert r == Block(
            instructions=[]
        )

        units = [
            lex.Unit('{', Symbols.LeftBracket),
            lex.Unit('{', Symbols.LeftBracket),
            lex.Unit('}', Symbols.RightBracket),
            lex.Unit('}', Symbols.RightBracket)
        ]

        r, offset, err = Block.parse(units)
        assert err is None
        assert offset == 4
        assert r == Block(
            instructions=[
                Instruction(value=Block(instructions=[]))
            ]
        )

        units = [
            lex.Unit('{', Symbols.LeftBracket),
            lex.Unit('i', Symbols.Ident),
            lex.Unit('{', Symbols.LeftBracket),
            lex.Unit('}', Symbols.RightBracket),
            lex.Unit('}', Symbols.RightBracket)
        ]

        r, offset, err = Block.parse(units)
        assert err is None
        assert offset == 5
        assert r == Block(
            instructions=[
                Instruction(value=lex.Unit('i', Symbols.Ident)),
                Instruction(value=Block(instructions=[]))
            ]
        )

    def test_advanced(self):

        SyntGeneric._templated = {}

        class Assignment(ASynt):
            name: Symbols.Ident
            _eq: Symbols.Equals
            value: 'Block'

        class Rhythm(ASynt):
            _left: Symbols.Slash
            value: Symbols.NumberLiteral

        class Note(ASynt):
            name: Symbols.Ident
            octave: Optional[Literal[Symbols.NumberLiteral]]
            alteration: Optional[SyntUnion[Symbols.Plus, Symbols.Dash]]
            rhythm: Optional[Rhythm]

        class Block(ASynt):
            _left: Symbols.LeftBracket
            instructions: Chain['Instruction']
            _right: Symbols.RightBracket
        
        class Instruction(ASynt):
            value: SyntUnion[Block, Symbols.Ident, Note]

        Block.resolve_forward_refs(locals())
        Assignment.resolve_forward_refs(locals())

        units = [
            lex.Unit('a', Symbols.Ident),
            lex.Unit('/', Symbols.Slash),
            lex.Unit('3', Symbols.NumberLiteral)
        ]

        r, offset, err = Note.parse(units)
        assert r == Note(
            name=lex.Unit('a', Symbols.Ident),
            rhythm=Rhythm(
                value=lex.Unit('3', Symbols.NumberLiteral)
            )
        )

        units = [
            lex.Unit('a', Symbols.Ident),
            lex.Unit('4', Symbols.NumberLiteral),
            lex.Unit('/', Symbols.Slash),
            lex.Unit('3', Symbols.NumberLiteral)
        ]

        r, offset, err = Note.parse(units)
        assert r == Note(
            name=lex.Unit('a', Symbols.Ident),
            octave=lex.Unit('4', Symbols.NumberLiteral),
            rhythm=Rhythm(
                value=lex.Unit('3', Symbols.NumberLiteral)
            )
        )
    
        units = [
            lex.Unit('a', Symbols.Ident),
            lex.Unit('4', Symbols.NumberLiteral),
            lex.Unit('+', Symbols.Plus),
            lex.Unit('/', Symbols.Slash),
            lex.Unit('3', Symbols.NumberLiteral)
        ]

        r, offset, err = Note.parse(units)
        assert r == Note(
            name=lex.Unit('a', Symbols.Ident),
            octave=lex.Unit('4', Symbols.NumberLiteral),
            alteration=lex.Unit('+', Symbols.Plus),
            rhythm=Rhythm(
                value=lex.Unit('3', Symbols.NumberLiteral)
            )
        )

        units = [
           lex.Unit('beep', Symbols.Ident),
           lex.Unit('=', Symbols.Equals),
           lex.Unit('{', Symbols.LeftBracket),
           lex.Unit('c', Symbols.Ident),
           lex.Unit('}', Symbols.RightBracket)
       ]

        res, offset, err = Assignment.parse(units)
        ab = Assignment(
            name=lex.Unit('beep', Symbols.Ident),
            value=Block(
                instructions=[Instruction(value=lex.Unit('c', Symbols.Ident))]
            )
        )
        assert res == ab
        assert offset == 5
        assert err is None
