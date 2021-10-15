from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Set, Tuple, Union


class Symbols(Enum):
    Ident = 'ident'
    LeftPar, RightPar = '(', ')'
    LeftBracket, RightBracket = '{', '}'
    LeftSquareBracket, RightSquareBracket = '[', ']'
    FuncKeyword = 'fn'
    Comma = ','
    NumberLiteral = r'\d+'


class Specials(Enum):
    Any = 'any'
    Empty = 'empty'


from unittest.mock import Mock


N = Mock(goal=None)


# FIXME add support for Empty and multiple children for same link

@dataclass
class Node:
    goal: Any = None  # if provided, should evaluate to True
    leaves: Set[Any] = field(default_factory=set)
    children: Dict[Union[Enum, Specials, None], 'Node'] = field(default_factory=dict)
    name: str = None

    def __rshift__(self, charge):
        return Rshifter(self, charge)

    def __hash__(self):
        return id(self)

    def first(self, items):
        return self.find(items, take_first=True)

    def find(self, items: List, take_first: bool = False) -> Tuple[bool, int]:
        skip = 1
        ans = False
        if not items:
            return (False, 0)
        first = items[0]
        if (take_first or not items[1:]) and (goal := (self.children.get(first, N).goal or self.children.get(Specials.Any, N).goal)):
            return (goal, 1)
        t = self.children.get(first)
        if t:
            ans, cskip = t.find(items[1:], take_first=take_first)
            skip += cskip
            return (ans, skip)

        candidates = []

        if Specials.Any in self.children:
            ans = take_first
            cans, cskip = self.children[Specials.Any].find(items[1:], take_first=take_first)
            if cskip:
                ans = cans
            candidates.append([cans, cskip])

        if Specials.Empty in self.children:
            cans, cskip = self.children[Specials.Empty].find(items[:], take_first=take_first)
            candidates.append([cans, cskip - 1])

        winner = None

        # what if there are two of them ? take shortest or longest ?
        if len(candidates) > 1:
            tiniest = None
            longest = None
            truecount = 0
            for c, size in candidates:
                if c:
                    truecount += 1
                    winner = (c, size)
                    if not tiniest or size < tiniest[1]:
                        tiniest = (c, size)
                    if not longest or size > longest[1]:
                        longest = (c, size)
            if truecount > 1:
                winner = tiniest if take_first else longest
            # FIXME
        else:
            winner = candidates and candidates[0]
        
        if winner:
            print(f'{winner = }, {items = }')
            cans, cskip = winner
            if cskip or cans:
                ans = cans 
                skip += cskip

        return (ans, skip)

    
    def __or__(self, node):
        return Node(children={Specials.Empty: [self, node]})

    def print(self, depth: int = 0, indent: int = 2, key: str = None, history: set = set()):
        if self in history:
            return print(' ' * depth * indent + (f'$ \033[96m{self.name}\033[m' if self.name else '...'))
        tstr = f'{key}: ' * bool(key) + (f'\033[1;92m{self.name or self.goal}\033[m' * bool(self.goal) or f'\033[93m{self.name}\033[m' * bool(self.name) or '$')
        print(' ' * depth * indent + tstr, '{' + '}' * (not len(self.children)))
        for key, value in self.children.items():
            value.print(depth + 1, key=key, history={*history, self})
        if len(self.children):
            print(' ' * depth * indent + '}')


class Rshifter:

    def __init__(self, left, op):
        self.left = left
        if not isinstance(op, (list, tuple)):
            op = [op]
        self.op = op

    def __rshift__(self, right):

        node = self.left
        for op in self.op[:-1]:
            child = node.children.get(op, Node())
            node.children[op] = child
            node = child
        op = self.op[-1]
        node.children[op] = right
        return right


class TestNode:

    def test_rshift(self):
        n = Node()
        m = Node()
        r = n >> None >> m
        assert r is m
        assert n.children[None] == m

    def test_rshift_n(self):
        n = Node()
        m = Node()
        r = n >> 'a' >> Node() >> 'b' >> m
        assert r is m
        assert n.children['a'].children['b'] == m

    def test_rshift_n_short(self):
        n = Node()
        m = Node()
        r = n >> ('a', 'b') >> m
        assert r is m

        n >> ('a', 'c') >> Node()
        assert n.children['a'].children.keys() == {'b', 'c'}

    def test_concrete(self):

        start = Node(name='start')
        end = Node(name='end', goal=True)
        item = Node(name='item')

        start >> Symbols.LeftPar >> item
        item  >> [Specials.Any, Symbols.Comma] >> item  # requires trailing comma always
        item  >> Symbols.RightPar >> end

        start.print()

        assert start.children == {
            Symbols.LeftPar: item,
        }

        assert item.children == {
            Specials.Any: Node(children={
                Symbols.Comma: item
            }),
            Symbols.RightPar: end
        }

        assert end.children == {}
    
    def test_concrete_oneliner(self):

        start = Node()
        end = Node(goal=True)
        item = Node()

        (
            start >> Symbols.LeftPar >> 
            item >> [Specials.Any, Symbols.Comma] >>
            item >> Symbols.RightPar >> end
        )

        start.print()
        
        assert start.children == {
            Symbols.LeftPar: item,
        }

        assert item.children == {
            Specials.Any: Node(children={
                Symbols.Comma: item
            }),
            Symbols.RightPar: end
        }

        assert end.children == {}

    class TestFind:

        class Expr:

            @classmethod
            def tree(cls):

                start = Node()
                end = Node(goal=cls)
                item = Node()

                start >> Symbols.LeftPar >> item
                item  >> [Specials.Any, Symbols.Comma] >> item
                item  >> Symbols.RightPar >> end

                return start

        def test_basic(self):
            
            units = [
                Symbols.LeftPar,
                Symbols.RightPar
            ]
            assert self.Expr.tree().find(units) == (self.Expr, 2)
        
        def test_extended(self):
    
            units = [
                Symbols.LeftPar,
                Symbols.Ident,
                Symbols.RightPar
            ]
            assert self.Expr.tree().find(units) == (False, 3)

            units = [
                Symbols.LeftPar,
                Symbols.Ident,
                Symbols.Comma,
                Symbols.RightPar
            ]
            assert self.Expr.tree().find(units) == (self.Expr, 4)
    
            units = [
                Symbols.LeftPar,
                Symbols.Ident,
                Symbols.Comma,
                Symbols.Ident,
                Symbols.Comma,
                Symbols.RightPar
            ]
            assert self.Expr.tree().find(units) == (self.Expr, 6)
    
    class TestAmbiguous:

        def test_ambiguous(self):

            # ( Symbols.Ident )
            class parexpr:
                start = Node()
                ( start
                >> [ Symbols.LeftPar, Symbols.Ident, Symbols.RightPar ] >> Node(goal='parexpr')
                )

            # ( Symbols.Ident, )
            class parlist:
                start = Node()
                item  = Node()
                ( start
                >> Symbols.LeftPar >> item
                    >> [ Symbols.Ident, Symbols.Comma ] >> item
                        >> Symbols.RightPar >> Node(goal='parlist')
                    >> Node(goal='parexpr')
                )

            units_parexpr = [
                Symbols.LeftPar, Symbols.Ident, Symbols.RightPar
            ]

            assert parexpr.start.find(units_parexpr) == ('parexpr', 3)

            units_parlist = [
                Symbols.LeftPar, Symbols.Ident, Symbols.Comma, Symbols.RightPar
            ]

            assert parlist.start.find(units_parlist) == ('parlist', 4)

            assert parexpr.start.find(units_parlist) == (False, 3)
            assert parlist.start.find(units_parexpr) == (False, 3)

            # assert (parexpr.start | parlist.start).find(units_parlist) == ('parlist', 4)
            # assert (parexpr.start | parlist.start).find(units_parexpr) == ('parexpr', 3)
