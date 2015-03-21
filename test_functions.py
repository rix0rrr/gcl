import unittest
from os import path

import gcl


class TestFunctions(unittest.TestCase):
  def testStringInterpolation(self):
    x = gcl.loads("""
    things = { foo = 'FOO'; bar = 'BAR' };
    y = fmt('Hey {foo}, are you calling me a {bar}?', things)
    """)
    self.assertEquals('Hey FOO, are you calling me a BAR?', x['y'])
