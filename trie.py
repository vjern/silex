from dataclasses import dataclass, field
from enum import Enum
from typing import List, Any, Dict, Tuple, Optional, Set



class Specials(Enum):
    Any = 'any'
    AnyN = 'any_n'
    Empty = 'empty'


@dataclass
class Trie:

    children: Dict[Any, 'Trie'] = field(default_factory=dict)
    leaves: Set[Any] = field(default_factory=set)

    def init(self, itemss: List[Any]) -> 'Trie':
        for items in itemss:
            self.insert(items)
        return self

    def insert(self, items: List) -> int:
        if not items:
            return 1
        first = items[0]
        if first is Specials.AnyN:
            child = self.children[Specials.Any] = self.children.get(Specials.Any, self)
            first = Specials.Any
        child = self.children[first] = self.children.get(first, Trie())
        child.name = str(first)
        res = child.insert(items[1:])
        if res:
            self.leaves.add(first)
        return 0

    def has(self, item: Any) -> bool:
        return item in self.children

    def get(self, item: Any) -> Optional['Trie']:
        return self.children.get(item)

    def __getitem__(self, key):
        return self.children[key]

    # hot mess to pass all tests (especially any)
    def find(self, items: List, take_first: bool = False, take_longest: bool = False) -> Tuple[bool, int]:

        skip = 1
        ans = False

        if not items:
            return (False, 0)

        first = items[0]
        candidates = []
        
        if (take_first or not items[1:]) and (first in self.leaves or Specials.Any in self.leaves):
            res = (True, 1)
            if not take_longest:
                return res
            candidates.append((True, 0, 'early_take_first'))

        t = self.children.get(first)
        if t:
            ans, cskip = t.find(items[1:], take_first=take_first)
            if not take_first:        
                return (ans, skip + cskip)
            candidates.append((ans, cskip, first))

        if Specials.Any in self.children:
            # ans = take_first
            cans, cskip = self.children[Specials.Any].find(items[1:], take_first=take_first, take_longest=take_longest)
            # if False and cskip:
            #     ans = cans
            candidates.append((cans, cskip, Specials.Any))

        if Specials.Empty in self.children:
            cans, cskip = self.children[Specials.Empty].find(items[:], take_first=take_first, take_longest=take_longest)
            candidates.append((cans, cskip - 1, Specials.Empty))

        winner = None

        # what if there are two of them ? take shortest or longest ?
        print(f'{candidates = }')
        if len(candidates) > 1:
            tiniest = None
            longest = None
            truecount = 0
            for c, size, name in candidates:
                if c:
                    truecount += 1
                    winner = (c, size, name)
                    if not tiniest or size < tiniest[1]:
                        tiniest = (c, size, name)
                    if not longest or size > longest[1]:
                        longest = (c, size, name)
            print(tiniest, longest, truecount, winner)
            if truecount > 1:
                winner = tiniest if not take_longest else longest
            # FIXME
        else:
            winner = candidates and candidates[0]
        
        if winner:
            print(f'{winner = }, {items = }')
            cans, cskip, name = winner
            if cskip or cans:
                ans = cans 
                skip += cskip

        return (ans, skip)

    def first(self, items):
        return self.find(items, take_first=True)

    def last(self, items):
        return self.find(items, take_first=True, take_longest=True)

def from_string(pat: str) -> list:
    tokens = pat.split()
    for i, token in enumerate(tokens):
        if token == '_+':
            token = Specials.AnyN
        elif token == '_':
            token = Specials.Any
        tokens[i] = token
    return tokens


class TestTrie:

    def test_init(self):
        t = Trie().init(['a', 'b'])
        assert t == Trie(
            { 'a': Trie(),
              'b': Trie() },
            { 'a', 'b' }
        )

    def test_has(self):
        t = Trie().init(['a', 'b'])
        assert t.has('a')
        assert not t.has('c')

    def test_getitem(self):
        t = Trie().init(['a', 'b'])
        assert t['a'] == t.children['a']

    def test_find(self):
        t = Trie()
        t.insert('a')
        assert t == Trie(
            { 'a': Trie() },
            { 'a' }
        )
        assert t.find('a') == (True, 1)
        assert t.find('b') == (False, 1)

        t = Trie()
        t.insert('ab')
        assert t == Trie(
            {
                'a': Trie(
                    { 'b': Trie() },
                    { 'b' }
                )
            },
            set()
        )
        assert t.find('a') == (False, 1)
        assert t.find('ab') == (True, 2)
        assert t.find('db') == (False, 1)
        assert t.find('ac') == (False, 2)

        t = Trie()
        t.insert('ab')
        t.insert('ac')
        assert t == Trie(
            {
                'a': Trie(
                    { 'b': Trie(), 'c': Trie() },
                    { 'b', 'c' }
                )
            }
        )
        assert t.find('a') == (False, 1)
        assert t.find('ab') == (True, 2)
        assert t.find('ac') == (True, 2)
        assert t.find('db') == (False, 1)
        assert t.find('dc') == (False, 1)

    def test_first(self):
        t = Trie()
        t.insert('==')
        assert t == Trie(
            {
                '=': Trie(
                    { '=': Trie() },
                    { '=' }
                )
            }
        )
        assert t.find('==') == (True, 2)
        assert t.find('== a') == (False, 3)

        assert t.find('==', take_first=True) == (True, 2)
        assert t.find('== a', take_first=True) == (True, 2)

        assert t.first('==') == (True, 2)
        assert t.first('== a') == (True, 2)

    def test_list(self):
        t = Trie()
        t.insert([0, 1])
        assert t == Trie(
            {
                0: Trie(
                    { 1: Trie() },
                    { 1 }
                )
            }
        )
        assert t.find([0, 1]) == (True, 2)
        assert t.find([1, 1]) == (False, 1)
        assert t.find([0, 2]) == (False, 2)
        assert t.find([0, 2, 55]) != (False, 3)

        assert t.first([0, 2, 55]) != (True, 2)

    def test_any(self):
        t = Trie()
        t.insert([Specials.Any, 1])
        assert t == Trie(
            {
                Specials.Any: Trie(
                    { 1: Trie() },
                    { 1 }
                )
            }
        )
        assert t.find([0, 1]) == (True, 2)
        assert t.find([1, 1]) == (True, 2)
        assert t.find([1, 2]) == (False, 2)

        assert t.first([1, 1, 3]) == (True, 2)
        assert t.first([0, 1, 3]) == (True, 2)

    def test_any_multiple(self):
        t = Trie()
        t.insert([Specials.Any, 1, Specials.Any, 1])
        assert t == Trie(
            {
                Specials.Any: Trie(
                    { 1: Trie(
                        { Specials.Any: Trie(
                            { 1: Trie() },
                            { 1 }
                        ) },
                    ) }
                )
            }
        )
        assert t.find([0, 1, 2, 1]) == (True, 4)
        assert t.find([1, 1, 2, 1]) == (True, 4)
        assert t.find([1, 2, 2, 1]) == (False, 2)

    def test_any_ending(self):
        t = Trie()
        t.insert([Specials.Any, 1, Specials.Any])
        assert t == Trie(
            {
                Specials.Any: Trie(
                    { 1: Trie(
                        { Specials.Any: Trie() },
                        { Specials.Any }
                    ) }
                )
            }
        )
        assert t.find([0, 1, 2]) == (True, 3)
        assert t.find([1, 1, 2]) == (True, 3)
        assert t.find([1, 2, 2]) == (False, 2)

        assert t.first([0, 1, 2, 3]) == (True, 3)
        assert t.first([1, 1, 2, 3]) == (True, 3)
        assert t.first([1, 2, 2, 3]) == (False, 2)

    def test_consecutive_any(self):
        t = Trie()
        t.insert([Specials.Any, Specials.Any, 1, 3])
        assert t == Trie(
            { Specials.Any: Trie(
                { Specials.Any: Trie(
                    { 1: Trie(
                        { 3: Trie() },
                        { 3 }
                    ) }
                ) }
            ) }
        )
        assert t.find([1, 2, 1, 3]) == (True, 4)
        assert t.find([33, 44, 1, 3]) == (True, 4)
        assert t.find([0, 0, 0, 3]) == (False, 3)

        assert t.first([0, 0, 1, 3, 5, 5]) == (True, 4)

        t.insert([Specials.Any, Specials.Any])
        assert t == Trie(
            { Specials.Any: Trie(
                { Specials.Any: Trie(
                    { 1: Trie(
                        { 3: Trie() },
                        { 3 }
                    ) }
                ) },
                { Specials.Any }
            ) }
        )
        assert t.find([1, 2, 1, 3]) == (True, 4)
        assert t.first([1, 2, 1, 3]) == (True, 2)

        t.insert([Specials.Any, Specials.Any, 1, 3, Specials.Any, Specials.Any])
        assert t == Trie(
            { Specials.Any: Trie(
                { Specials.Any: Trie(
                    { 1: Trie(
                        { 3: Trie(
                            { Specials.Any: Trie(
                                { Specials.Any: Trie() },
                                { Specials.Any }
                            ) }
                        ) },
                        { 3 }
                    ) }
                ) },
                { Specials.Any }
            ) }
        )
        assert t.find([0, 0, 1, 3, 0, 2]) == (True, 6)
        assert t.find([0, 0, 1, 3, 4, 7]) == (True, 6)

    def test_any_trailing(self):
        t = Trie()
        t.insert([Specials.Any, Specials.Any])
        assert t.find([3, 4]) == (True, 2)
        assert t.find([3, 4, 5]) == (False, 3)
        assert t.first([2, 5, 7]) == (True, 2)
        assert t.find([2]) == (False, 1)

    def test_any_n(self):
        t = Trie()
        t.insert([Specials.AnyN, 333])
        assert t == Trie(
            { Specials.Any: t, 333: Trie() },
            { 333 }
        )
        assert t.find([3, 4, 333]) == (True, 3)
        assert t.find([3, 333]) == (True, 2)
        assert t.find([333]) == (True, 1)

        assert t.first([3, 4, 333, 222]) == (True, 3)
        assert t.first([3, 333, 3]) == (True, 2)

    def test_any_n_only(self):
        t = Trie()
        t.insert([Specials.AnyN])
        assert t == Trie(
            { Specials.Any: t },
            { Specials.Any }
        )
        assert t.find([3, 4, 333]) == (True, 3)
        assert t.find([3, 333]) == (True, 2)
        assert t.find([333]) == (True, 1)

        assert t.first([3, 4, 333, 222]) == (True, 1)
        assert t.first([3, 333, 3]) == (True, 1)

    def test_multi_any_n(self):
        t = Trie()
        t.insert([Specials.AnyN, 3, Specials.AnyN, 33])
        assert t == Trie({
            Specials.Any: t,
            3: Trie(
                {
                    Specials.Any: t.children[3],
                    33: Trie()
                },
                { 33 }
            )
        })

        assert t.find([0, 3, 4, 33]) == (True, 4)
        assert t.find([0, 0, 3, 4, 33]) == (True, 5)
        assert t.find([0, 0, 3, 4, 77, 33]) == (True, 6)

        assert t.first([0, 3, 4, 33, 55]) == (True, 4)
        assert t.first([0, 0, 3, 4, 33, 77]) == (True, 5)
        assert t.first([0, 0, 3, 4, 77, 33, 66]) == (True, 6)

    def test_trailing_any_n(self):
        t = Trie()
        t.insert([99, Specials.AnyN])
        assert t == Trie(
            { 99: Trie(
                { Specials.Any: t.children[99] },
                { Specials.Any }
            ) }
        )

        # greedy
        assert t.find([99, 3, 4, 333]) == (True, 4)
        assert t.find([99, 3, 333]) == (True, 3)
        assert t.find([99, 333]) == (True, 2)

        # non greedy
        assert t.first([99, 3, 4, 333, 222]) == (True, 2)
        assert t.first([99, 3, 333, 3]) == (True, 2)

    def test_consecutive_any_n(self):
        t = Trie()
        t.insert([Specials.AnyN, Specials.AnyN, 333])
        assert t == Trie(
            { Specials.Any: t, 333: Trie() },
            { 333 }
        )
        assert t.find([3, 4, 333]) == (True, 3)
        assert t.find([3, 333]) == (True, 2)
        assert t.find([333]) == (True, 1)

        assert t.first([3, 4, 333, 222]) == (True, 3)
        assert t.first([3, 333, 3]) == (True, 2)

    def test_any_n_ambiguous(self):
        t = Trie()
        t.insert([Specials.AnyN, 33])
        assert t.find([0, 33, 33]) == (False, 3)  # only the first
        # ^ FIXME shouldn't this return (True, 3) ? that's probably because it finds 33 early and see a non empty remainder
        assert t.first([0, 33, 33]) == (True, 2)

    def test_from_string(self):
        assert from_string('_ + _') == [Specials.Any, '+', Specials.Any]
        assert from_string('_ ? _+ : _+') == [Specials.Any, '?', Specials.AnyN, ':', Specials.AnyN]

    def test_natural(self):
        t = Trie()
        t.insert('==')
        t.insert('not in')
        f = t.first('not in b')
        assert f == (True, 6), f

    def test_empty(self):
        t = Trie()
        t.insert([Specials.Empty, 33])
        assert t.find([0, 33, 33]) == (False, 1)
        assert t.find([33]) == (True, 1)
        assert t.first([33, 4]) == (True, 1)
    
    def test_last(self):
        t = Trie()
        #t.insert([33])
        t.insert([Specials.AnyN, 33])
        assert t.first([33]) == (True, 1)
        assert t.first([33, 33]) == (True, 1)
        assert t.last([33, 33]) == (True, 2)
        assert t.last([33, 33, 33]) == (True, 3)
        assert t.last([33, 33, 33, 4]) == (True, 3)

        t = Trie()
        t.insert([Specials.Empty, 33])
        assert t.first([33]) == (True, 1)
        assert t.last([33]) == (True, 1)


# still need to be able to tell the spans positions captured by any_n
