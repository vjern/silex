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


Empty = Specials.Empty


from unittest.mock import Mock


N = Mock(goal=None)


# FIXME add support for Empty and multiple children for same link

@dataclass
class Node:
    name: str = None
    goal: Any = None  # if provided, should evaluate to True
    children: Dict[Union[Enum, Specials, None], 'Node'] = field(default_factory=dict)

    @classmethod
    def make(cls, n: int):
        import string
        return [cls(name=l) for l in string.ascii_lowercase[:n]]

    def __eq__(self, node):
        return self.goal == node.goal and self.children == node.children

    def __rshift__(self, charge):
        return Rshifter(self, charge)

    def __hash__(self):
        return id(self)

    def first(self, items):
        return self.find(items, take_first=True)
    
    def last(self, items):
        return self.find(items, take_first=True, take_longest=True)
    
    def assert_equals(self, node, history: set = set()):
        if self in history:
            return True
        if node is self:
            return True
        assert self.goal == node.goal, (self.goal, node.goal, self, node)
        assert self.children.keys() == node.children.keys(), (self.children.keys(), node.children.keys())
        assert all(
            self.children[key].assert_equals(node.children[key], {*history, self})
            for key in self.children.keys()
        )
        return True

    def find(self, items: List, take_first: bool = False, take_longest: bool = False) -> Tuple[bool, int]:

        skip = 1
        ans = False

        if not items:
            return False, 0

        first = items[0]
        candidates = []
        
        if (take_first or not items[1:]) and (goal := (self.children.get(first, N).goal or self.children.get(Specials.Any, N).goal)):
            if not take_longest:
                return goal, 1
            candidates.append((goal, 0, 'early_take_first'))

        t = self.children.get(first)
        if t:
            ans, cskip = t.find(items[1:], take_first=take_first, take_longest=take_longest)
            if not take_first:        
                return (ans, skip + cskip)
            candidates.append((ans, cskip, first))

        t = self.children.get(Specials.Any)
        if t:
            cans, cskip = t.find(items[1:], take_first=take_first, take_longest=take_longest)
            candidates.append((cans, cskip, Specials.Any))

        t = self.children.get(Specials.Empty)
        if t:
            cans, cskip = t.find(items[:], take_first=take_first, take_longest=take_longest)
            candidates.append((cans, cskip - 1, Specials.Empty))

        winner = None

        if len(candidates) > 1:
            shortest = None
            longest = None
            for c, size, name in candidates:
                if c:
                    winner = (c, size, name)
                    if not shortest or size < shortest[1]:
                        shortest = (c, size, name)
                    if not longest or size > longest[1]:
                        longest = (c, size, name)
            winner = longest if take_longest else shortest
        else:
            winner = candidates and candidates[0] or None
        
        if winner:
            cans, cskip, name = winner
            if cskip or cans:
                ans = cans
                skip += cskip

        return ans, skip

    def __or__(self, node):
        # return Node(children={Specials.Empty: [self, node]})
        return self.merge(node)

    def print(self, depth: int = 0, indent: int = 2, key: str = None, history: set = set()):
        if self in history:
            return print(' ' * depth * indent + (f'{key}: \033[96m{self.name}\033[m' if self.name else '...'))
        tstr = f'{key}: ' * bool(key) + (f'\033[1;92m{self.name} = {self.goal}\033[m' * bool(self.goal) or f'\033[93m{self.name}\033[m' * bool(self.name) or '$')
        print(' ' * depth * indent + tstr, '{' + '}' * (not len(self.children)))
        for key, value in self.children.items():
            value.print(depth + 1, key=key, history={*history, self})
        if len(self.children):
            print(' ' * depth * indent + '}')
    
    def merge(self, node, inplace: bool = False):
        n = Node()
        if inplace:
            n = self
        if self.goal and node.goal and self.goal != node.goal:
            raise Exception('Conflicting goals %s %s' % (self.goal, node.goal))
        n.goal = self.goal or node.goal
        n.children = {**self.children, **node.children}
        for common in self.children.keys() & node.children.keys():
            n.children[common] = self.children[common].merge(node.children[common], inplace=inplace)
        return n


class Rshifter:

    def __init__(self, left, op):
        self.left = left
        if not isinstance(op, (list, tuple)):
            op = [op]
        self.op = op

    def __rshift__(self, right):
        if not self.op:
            self.op = [Specials.Empty]
        node = self.left
        for op in self.op[:-1]:
            child = node.children.get(op, Node())
            node.children[op] = child
            node = child
        op = self.op[-1]
        r = node.children.get(op)
        if r and r is not right:
            if not right.goal:
                right.goal = r.goal
            assert not right.children.keys() & r.children.keys(), 'Cannot merge nodes with overlapping edges'
            right.children.update(r.children)
        node.children[op] = right  # this is the culprit; it doesn't check if op is already in children
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

    def test_composite(self):
        
        pats = [
            [Symbols.Ident, Symbols.NumberLiteral],
            [Symbols.NumberLiteral],
            [Symbols.Ident],
        ]

        a = Node()

        for p in pats:
            a >> p >> Node(goal=Node)
            print(p)
            a.print()
        
        a.print()
        assert a.children[Symbols.Ident].children.keys() == {Symbols.NumberLiteral}


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

    class TestMerge:

        def test_basic(self):
            a = Node(goal=int)
            b = Node()
            n = Node.merge(a, b)
            assert n.goal is int
            assert n.children == {}
        
        def test_simple(self):
            
            a, b, c = Node.make(3)
            a.goal = int
            c.goal = str
            a.children = { 1: b, 2: c }

            d, e = Node.make(2)
            e.goal = int
            d.children = { 3: e }

            n = a.merge(d)

            assert n.children == {
                1: b, 2: c, 3: e
            }
        
        def test_overlap(self):
            
            a, b, c = Node.make(3)
            a.goal = int
            c.goal = str
            a.children = { 1: b, 2: c }

            d, e, f = Node.make(3)
            f.goal = int
            d.children = { 2: e }
            e.children = { 3: f }

            # c & e get merged
            n = a.merge(d)

            assert n.children == {
                1: b,
                2: Node(
                    goal=str,
                    children={
                        3: f
                })
            }
