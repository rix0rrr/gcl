import unittest

import gcl

def parse(s, env=None):
  return gcl.loads(s).eval(env or {})

class TestBasics(unittest.TestCase):
  def testInteger(self):
    self.assertEquals(3, parse('3'))

  def testNegativeInteger(self):
    self.assertEquals(-3, parse('-3'))

  def testFloat(self):
    self.assertEquals(3.14, parse('3.14'))
    self.assertEquals(0.14, parse('.14'))

  def testSingleQuotedString(self):
    self.assertEquals("foo", parse("'foo'"))

  def testDoubleQuotedString(self):
    self.assertEquals("foo", parse('"foo"'))

  def testBool(self):
    self.assertEquals(True, parse('true'))
    self.assertEquals(False, parse('false'))

  def testComments(self):
    self.assertEquals(3, parse("""
      # comment
      3
      """))

  def testComments(self):
    self.assertEquals(3, parse("""
      3
      # comment"""))


class TestList(unittest.TestCase):
  def testEmptyList(self):
    self.assertEquals([], parse('[]'))

  def testSingletonList(self):
    self.assertEquals([1], parse('[1]'))

  def testTrailingComma(self):
    self.assertEquals([1], parse('[1, ]'))

  def testPair(self):
    self.assertEquals([1, 2], parse('[1, 2]'))

  def testNestedList(self):
    self.assertEquals([1, [2]], parse('[1, [2]]'))


class TestVariable(unittest.TestCase):
  def testVariableWithEnv(self):
    self.assertEquals('bar', parse('foo', env={'foo': 'bar'}))


class TestTuple(unittest.TestCase):
  def testEmptyTuple(self):
    self.assertEquals([], parse('{}').items())

  def testBoundIdentifiers(self):
    self.assertEquals([('foo', 'bar')], parse('{ foo = "bar" }').items())

  def testUnboundIdentifiers(self):
    t = parse('{ foo; }')
    try:
      print t['foo']
      self.fail('Should have thrown')
    except LookupError:
      pass  # Expected

  def testIndirectUnbound(self):
    t = parse('{ foo; bar = foo + 3; }')
    try:
      print t['bar']
      self.fail('Should have thrown')
    except LookupError:
      pass  # Expected

  def testVariableInSameScope(self):
    t = parse('{ foo = 3; bar = foo; }')
    self.assertEquals(3, t['bar'])

  def testVariableFromParentScope(self):
    t = parse('{ foo = global; bar = foo; }', { 'global': 3 })
    self.assertEquals(3, t['bar'])

  def testShadowGlobalScope(self):
    t = parse('{ foo = 3; bar = foo; }', { 'foo': 2 })
    self.assertEquals(3, t['bar'])

  def testDontInheritFromEnclosingTuple(self):
    # We don't want the child to implicitly inherit from the parent
    # scope--that'll lead to an unmaintainable mess, especially across
    # includes.
    t = parse("""{
      foo = 3;
      child = {
        bar = foo;
      }
    }""", { 'foo': 2 })
    self.assertEquals(2, t['child']['bar'])

  def testDereferencing(self):
    pass


class TestApplication(unittest.TestCase):
  def setUp(self):
    env = {}
    env['inc'] = lambda x: x + 1
    env['foo'] = lambda: lambda: 3
    env['add'] = lambda x, y: x + y
    env['curry_add'] = lambda x: lambda y: x + y
    self.env = gcl.Environment(env)

  def testFunctionApplication(self):
    self.assertEquals(3, parse('inc(2)', env=self.env))

  def testFunctionApplicationMultiArgs(self):
    self.assertEquals(5, parse('add(2, 3)', env=self.env))

  def testRepeatedFunctionCalls(self):
    self.assertEquals(3, parse('foo()()', env=self.env))

  def testNestedFunctionCalls(self):
    self.assertEquals(4, parse('inc(inc(2))', env=self.env))

  def testFunctionApplicationWithoutParens(self):
    self.assertEquals(3, parse('inc 2', env=self.env))

  def testChainedApplicationWithoutParens(self):
    self.assertEquals(5, parse('curry_add 2 3', env=self.env))

  def testTupleComposition(self):
    pass

  def testDereferencingFromFunctionCall(self):
    pass


class TestExpressions(unittest.TestCase):
  def testAdd(self):
    self.assertEquals(5, parse('2 + 3'))

  def testAddStrings(self):
    self.assertEquals('foobar', parse('"foo" + "bar"'))

  def testMul(self):
    self.assertEquals(6, parse('2 * 3'))
