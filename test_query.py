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
    self.assertEquals(set([3, 4]), set(sel.select(self.model).values()))

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


class TestFinder(unittest.TestCase):
  def setUp(self):
    self.obj = gcl.loads('''
      Thing = {
        type = 'thing';
      };

      some-stuff = {
        thing1 = Thing;
        thing2 = Thing { something = 'different'; parent = thing1 };
      };

      heres_a_thing = Thing { type = 'boo'; after = some-stuff.thing2 };

      in_list = [Thing];

      deep_scope = {
        in_list = [{ type = 'nothing';  }, { kind = 'nice' }];
      };
      ''')

  def testFindObjectsNotInList(self):
    finder = query.TupleFinder(query.HasKeyCondition('type'))
    finder.find(self.obj)
    self.assertEquals(3, len(set(finder.unordered)))

  def testFindObjectsAlsoInList(self):
    finder = query.TupleFinder(query.HasKeyCondition('type', search_lists=True))
    finder.find(self.obj)
    self.assertEquals(5, len(set(finder.unordered)))

  def testFindInList(self):
    finder = query.TupleFinder(query.InListCondition('in_list'));
    finder.find(self.obj)
    self.assertEquals(3, len(set(finder.unordered)))

  def testOrderDependencies(self):
    finder = query.TupleFinder(query.HasKeyCondition('type'))
    finder.find(self.obj)
    finder.order()
    self.assertEquals([], finder.unordered)
    self.assertFalse(finder.has_recursive_dependency())
    self.assertEquals(['thing', 'thing', 'boo'], [x.value['type'] for x in finder.ordered])

  def testRecursiveDependencies(self):
    self.obj = gcl.loads('''
      Thing = {
        type = 'thing';
      };

      thing1 = Thing {
        friend = thing2;
      };

      thing2 = Thing {
        friend = thing1;
      }
      ''')
    finder = query.TupleFinder(query.HasKeyCondition('type'))
    finder.find(self.obj)
    finder.order()
    self.assertTrue(finder.has_recursive_dependency())

    recursive_names = [n.key_name() for n in finder.find_recursive_dependency()]
    self.assertEquals(['thing1', 'thing2', 'thing1'], recursive_names)

