from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Set, Tuple, Union

from g import Symbols


class Specials(Enum):
    Any = 'any'
    Empty = 'empty'


from unittest.mock import Mock


N = Mock(goal=None)


@dataclass
class Node:
    goal: Any = None
    leaves: Set[Any] = field(default_factory=set)
    children: Dict[Union[Enum, Specials, None], 'Node'] = field(default_factory=dict)

    def __rshift__(self, charge):
        return Rshifter(self, charge)

    def find(self, items: List, take_first: bool = False) -> Tuple[bool, int]:
        skip = 1
        ans = False
        if not items:
            return (False, 0)
        first = items[0]
        if (take_first or not items[1:]) and (self.children.get(first, N).goal or self.children.get(Specials.Any, N).goal):
            return (True, 1)
        t = self.children.get(first)
        if t:
            ans, cskip = t.find(items[1:], take_first=take_first)
            skip += cskip
        elif Specials.Any in self.children:
            ans = take_first
            cans, cskip = self.children[Specials.Any].find(items[1:], take_first=take_first)
            if cskip:
                ans = cans
            skip += cskip
        return (ans, skip)


class Rshifter:

    def __init__(self, left, op):
        self.left = left
        if not isinstance(op, (list, tuple)):
            op = [op]
        self.op = op

    def __rshift__(self, right):
        stack = [
            self.left,
            *[Node() for _ in range(len(self.op) - 1)],
            right
        ]
        op_stack = list(reversed(self.op))
        left = stack[0]
        for right in stack[1:]:
            left.children[op_stack.pop()] = right
            left = right
        return left


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

    def test_concrete(self):

        start = Node()
        end = Node(goal=True)
        item = Node()

        start >> Symbols.LeftPar >> item
        item  >> [Specials.Any, Symbols.Comma] >> item  # requires trailing comma always
        item  >> Symbols.RightPar >> end

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

        start = Node()
        end = Node(goal=True)
        item = Node()

        start >> Symbols.LeftPar >> item
        item  >> [Specials.Any, Symbols.Comma] >> item
        item  >> Symbols.RightPar >> end

        def test_basic(self):
            
            units = [
                Symbols.LeftPar,
                Symbols.RightPar
            ]
            assert self.start.find(units) == (True, 2)
        
        def test_extended(self):
    
            units = [
                Symbols.LeftPar,
                Symbols.Ident,
                Symbols.RightPar
            ]
            assert self.start.find(units) == (False, 3)

            units = [
                Symbols.LeftPar,
                Symbols.Ident,
                Symbols.Comma,
                Symbols.RightPar
            ]
            assert self.start.find(units) == (True, 4)
    
            units = [
                Symbols.LeftPar,
                Symbols.Ident,
                Symbols.Comma,
                Symbols.Ident,
                Symbols.Comma,
                Symbols.RightPar
            ]
            assert self.start.find(units) == (True, 6)
    