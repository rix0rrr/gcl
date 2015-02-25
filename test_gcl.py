import unittest

import gcl

def parse_ast(s, implicit_tuple=False):
  return gcl.reads(s, implicit_tuple=implicit_tuple)

def parse(s, env=None, implicit_tuple=False):
  return (gcl.reads(s, implicit_tuple=implicit_tuple)
             .eval(gcl.default_env.extend(env)))

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

  def testNull(self):
    self.assertEquals(None, parse('null'))

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

  def testImplicitTupleBasic(self):
    self.assertEquals([('foo', 3)], parse('foo=3;', implicit_tuple=True).items())

  def testImplicitTupleEmpty(self):
    self.assertEquals([], parse('', implicit_tuple=True).items())


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
      print(t['foo'])
      self.fail('Should have thrown')
    except LookupError:
      pass  # Expected

  def testIndirectUnbound(self):
    t = parse('{ foo; bar = foo + 3; }')
    try:
      print(t['bar'])
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
    t = parse("""{
      obj = {
        attr = 1;
      };
      one = obj.attr;
    }""");
    self.assertEquals(1, t['one'])

  def testDereferencingFromEnvironment(self):
    x = parse('obj.attr', env={ 'obj': { 'attr' : 1 }})
    self.assertEquals(1, x)


class TestApplication(unittest.TestCase):
  def setUp(self):
    self.env = {}
    self.env['inc'] = lambda x: x + 1
    self.env['foo'] = lambda: lambda: 3
    self.env['add'] = lambda x, y: x + y
    self.env['curry_add'] = lambda x: lambda y: x + y
    self.env['mk_obj'] = lambda x: { 'attr': x }

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
    t = parse('{ foo = 1 } { bar = 2 }')
    self.assertEquals([
      ('bar', 2),
      ('foo', 1),
      ], sorted(t.items()))

  def testIndirectTupleComposition(self):
    t = parse("""
    {
      base = {
        name;
        hello = 'hello ' + name;
      };

      mine = base { name = 'Johnny' }
    }
    """)
    self.assertEquals('hello Johnny', t['mine']['hello'])

  def testDereferencingFromFunctionCall1(self):
    self.assertEquals(10, parse('mk_obj(10).attr', env=self.env))

  def testDereferencingFromFunctionCall2(self):
    self.assertEquals(10, parse('(mk_obj 10).attr', env=self.env))


class TestExpressions(unittest.TestCase):
  def testAdd(self):
    self.assertEquals(5, parse('2 + 3'))

  def testAddStrings(self):
    self.assertEquals('foobar', parse('"foo" + "bar"'))

  def testMul(self):
    self.assertEquals(6, parse('2 * 3'))

  def testPrecedence(self):
    self.assertEqual(10, parse('2 * 3 + 4'))

  def testNot(self):
    self.assertEqual(False, parse('not true'))

  def testLogicalOpsAndPrecedence(self):
    self.assertEquals(True, parse('0 < 5 and 2 <= 2'))

  def testConditional(self):
    self.assertEquals(1, parse('if 0 < 5 then 1 else 2'))
    self.assertEquals(2, parse('if 5 < 0 then 1 else 2'))


class TestStandardLibrary(unittest.TestCase):
  def testPathJoin(self):
    self.assertEquals('a/b/c', parse("path_join('a', 'b', 'c')"))


class TestIncludes(unittest.TestCase):
  def parse(self, fname, s):
    return gcl.loads(s, filename=fname, loader=self.loader).eval(gcl.default_env)

  def loader(self, base, rel):
    target_path = gcl.find_relative(base, rel)
    return gcl.loads('loaded_from = %r' % target_path).eval(gcl.default_env)

  def testRelativeInclude(self):
    t = self.parse('/home/me/file', 'inc = include "other_file"')
    self.assertEquals('/home/me/other_file', t['inc']['loaded_from'])

  def testRelativeIncludeUp(self):
    t = self.parse('/home/me/file', 'inc = include "../other_file"')
    self.assertEquals('/home/other_file', t['inc']['loaded_from'])

  def testAbsoluteInclude(self):
    t = self.parse('/home/me/file', 'inc = include "/other_file"')
    self.assertEquals('/other_file', t['inc']['loaded_from'])
