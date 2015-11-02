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
