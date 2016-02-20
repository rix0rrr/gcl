import unittest
from os import path

import gcl


class TestStringInterpolation(unittest.TestCase):
  def testStringInterpolation(self):
    x = gcl.loads("""
    things = { foo = 'FOO'; bar = 'BAR' };
    y = fmt('Hey {foo}, are you calling me a {bar}?', things)
    """)
    self.assertEquals('Hey FOO, are you calling me a BAR?', x['y'])

  def testLazyStringInterpolation(self):
    x = gcl.loads("""
    things = { foo = 'FOO'; boom; };
    y = fmt('Hi {foo}', things)
    """)
    self.assertEquals('Hi FOO', x['y'])

  def testSubInterpolation(self):
    x = gcl.loads("""
    things = { sub = { foo = 'FOO'; boom } };
    y = fmt('Hi {sub.foo}', things)
    """)
    self.assertEquals('Hi FOO', x['y'])

  def testImplicitScope(self):
    x = gcl.loads("""
    things = { foo = 'FOO'; boom };
    y = fmt 'Hi {things.foo}'
    """)
    self.assertEquals('Hi FOO', x['y'])

  def testJoin(self):
    x = gcl.loads("""
    things = join(['a', 'b', 'c']);
    """)
    self.assertEquals('a b c', x['things'])

  def testComposeAll(self):
    x = gcl.loads('''
    empty = compose_all([]);
    combined = compose_all([{ a = 'a' }, { b = 'b' }]);
    ''')
    self.assertEquals([], list(x['empty'].keys()))
    self.assertEquals(['a', 'b'], sorted(x['combined'].keys()))

  def testSorted(self):
    x = gcl.loads('''
      tup = { a = 1; b = 2; c = 3; d = 4 };
      keys = sorted([k for k in tup ])
    ''')
    self.assertEquals(['a', 'b', 'c', 'd'], x['keys'])

  def testSplit(self):
    x = gcl.loads('''
      list1 = split "one two three";
      list2 = split("one/two/three", "/");
    ''')

    self.assertEquals(['one', 'two', 'three'], x['list1'])
    self.assertEquals(['one', 'two', 'three'], x['list2'])

  def testHasKey(self):
    x = gcl.loads('''
    X = { one; two = 2; };
    has_one = has(X, 'one');
    has_two = has(X, 'two');
    has_three = has(X, 'three');
    ''')

    self.assertEquals(False, x['has_one'])
    self.assertEquals(True, x['has_two'])
    self.assertEquals(False, x['has_three'])

  def testHasKeyCompound(self):
    x = gcl.loads('''
    X = { key };
    Y = X { key = 3 };
    x_key = has(X, 'key');
    y_key = has(Y, 'key');
    ''')

    self.assertEquals(False, x['x_key'])
    self.assertEquals(True, x['y_key'])

  def testFlatten(self):
    x = gcl.loads('''
    list = flatten [[1], [2], [], [3, 4, 5], [[6]]];
    ''')

    self.assertEquals([1, 2, 3, 4, 5, [6]], x['list'])

  def testFlattenNotAccidentallyOnStrings(self):
    x = gcl.loads('''
    list = flatten ["abc"]
    ''')

    with self.assertRaises(ValueError):
      print(x['list'])
