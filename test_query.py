import unittest

import gcl
from gcl import query

class QueryTests(unittest.TestCase):
  def setUp(self):
    self.model = gcl.loads("""
    x = {
      y = { z = 3 };
      q = { z = 4 };
    };
    obj_list = [{x = 1}, {x = 2}, {x = 3}];
    scalar_list = [1, 2, 3];
    """)

  def testSimpleParse(self):
    sel = query.GPath('x.y.z')
    self.assertEquals(3, sel.select(self.model).first())

  def testSimpleParseDeep(self):
    sel = query.GPath('x.y.z')
    deep = sel.select(self.model).deep()
    self.assertEquals({'x': {'y': {'z': 3}}}, deep)

  def testStarParse(self):
    sel = query.GPath('x.*.z')
    self.assertEquals([3, 4], list(sel.select(self.model).values()))

  def testMultiParse(self):
    sel = query.GPath('x.{y,q}.z')
    self.assertEquals([3, 4], list(sel.select(self.model).values()))

  def testMultiSelectors(self):
    sel = query.GPath(['x.y.z', 'x.q.z'])
    self.assertEquals([3, 4], list(sel.select(self.model).values()))

  def testPathSelectors(self):
    sel = query.GPath(['x.y.z', 'x.q.z'])
    self.assertEquals([
      (['x', 'y', 'z'], 3),
      (['x', 'q', 'z'], 4),
      ], sel.select(self.model).paths_values())

  def testListSelectorsSub(self):
    sel = query.GPath('obj_list.*.x')
    self.assertEquals([1, 2, 3], list(sel.select(self.model).values()))

  def testListSelectors(self):
    sel = query.GPath('obj_list.*')
    result = sel.select(self.model).deep()
    self.assertTrue(isinstance(result['obj_list'], list))

  def testListIndex(self):
    sel = query.GPath('scalar_list.[1]')
    result = sel.select(self.model).first()
    self.assertEquals(2, result)

  def testListIndexDeep(self):
    sel = query.GPath('scalar_list.[1]')
    result = sel.select(self.model).deep()
    self.assertEquals([2], result['scalar_list'])

