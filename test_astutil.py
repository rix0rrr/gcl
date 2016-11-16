import unittest

import pyparsing

import gcl
from gcl import ast_util
from gcl import ast

class TestEnumerateScope(unittest.TestCase):
  def testSiblingsInScope(self):
    scope = readAndQueryScope('henk = 5', 1, 1)
    assert 'henk' in scope

  def testScopeObjectHasLocation(self):
    scope = readAndQueryScope('henk = 5', 1, 1)
    assert isinstance(scope['henk'], gcl.ast.TupleMemberNode)
    self.assertEquals(1, scope['henk'].location.lineno)
    self.assertEquals(1, scope['henk'].location.col)

  def testInherited(self):
    source = """
    henk = 5;
    other = {
      foo = 1;
    }
    """
    scope = readAndQueryScope(source, 3, 8) # foo
    assert 'henk' in scope
    assert 'foo' in scope

  def testCompositeNeg(self):
    source = """
    A = { henk = 5 };
    B = A { foo = 1 };
    """
    scope = readAndQueryScope(source, 2, 14)  # foo
    assert 'henk' not in scope
    assert 'foo' in scope

  def testCompositeMixin(self):
    source = """
    A = { henk = 5 };
    B = A { henk; foo = 1 };
    """
    scope = readAndQueryScope(source, 2, 14)  # foo
    assert 'henk' in scope
    assert 'foo' in scope

  def testTupleWithInherits(self):
    source = """
    henk = 5;
    B = { inherit henk }
    """
    scope = readAndQueryScope(source, 2, 14)  # inner tuple
    assert 'henk' in scope
    assert scope['henk'].location.lineno == 2  # Find the right henk

  def testIncludeDefaultBuiltins(self):
    tree = gcl.reads('henk = 5', filename='input.gcl')
    rootpath = tree.find_tokens(gcl.SourceQuery('input.gcl', 1, 1))
    return ast_util.enumerate_scope(rootpath, include_default_builtins=True)
    assert '+' in scope


class TestBrokenParseRecovery(unittest.TestCase):
  def testUnparseableValue(self):
    scope = readAndQueryScope("""
    outer = 1;
    tup = {
      pre_inner = 2;
      broken = + + ax 89sdf/b8;
      post_inner = 3;
    }
    """, 4, 10, allow_errors=True)
    self.assertSetEqual(set(['outer', 'tup', 'pre_inner', 'broken', 'post_inner']), set(scope.keys()))

  def testUnparseableTupleLineMissingSemicolon(self):
    scope = readAndQueryScope("""
    outer = 1;
    tup = {
      pre_inner = 2;
      broken = + + ax 89sdf/b8
      post_inner = 3;
    }
    """, 4, 10, allow_errors=True)
    self.assertSetEqual(set(['outer', 'tup', 'pre_inner', 'broken']), set(scope.keys()))

  def testUnparseableTupleLineCompleteGarbage(self):
    scope = readAndQueryScope("""
    outer = 1;
    tup = {
      pre_inner = 2;
      + + ax 89sdf/b8;
      post_inner = 3;
    }
    """, 4, 10, allow_errors=True)
    self.assertSetEqual(set(['outer', 'tup', 'pre_inner', 'post_inner']), set(scope.keys()))

  def testMissingEqualsRecover(self):
    scope = readAndQueryScope("""
      foo = 3;

      bar = {
          bippety bop;
      };
      """, 4, 11, allow_errors=True)
    self.assertSetEqual(set(['foo', 'bar', 'bippety']), set(scope.keys()))

  def testTrailingWordRecover(self):
    scope = readAndQueryScope("""
      foo = 3;
      f
      """, 2, 1, allow_errors=True)
    self.assertSetEqual(set(['foo', 'f']), set(scope.keys()))

  def testMissingClosingBraceRecover(self):
    scope = readAndQueryScope("""
    outer = 1;
    tup = {
      inner = 2;
    """, 3, 10, allow_errors=True)
    self.assertSetEqual(set(['outer', 'inner', 'tup']), set(scope.keys()))

  def testMissingClosingBraceRecoverDouble(self):
    scope = readAndQueryScope("""
    outer = 1;
    tup = {
      tap = {
        inner = 2;
    };
    """, 4, 10, allow_errors=True)
    self.assertSetEqual(set(['outer', 'inner', 'tup', 'tap']), set(scope.keys()))

  def testRecoverDoubleDefinition(self):
    scope = readAndQueryScope("""
    foo = 1;
    foo = 2;
    """, 2, 1, allow_errors=True)
    self.assertSetEqual(set(['foo']), set(scope.keys()))

  def testRecoverIncompleteDeref(self):
    scope = readAndQueryScope("""
    foo = bar.
    """, 1, 1, allow_errors=True)
    self.assertSetEqual(set(['foo']), set(scope.keys()))



def readAndQueryScope(source, line, col, **kwargs):
    tree = gcl.reads(source.strip(), filename='input.gcl', **kwargs)
    rootpath = tree.find_tokens(gcl.SourceQuery('input.gcl', line, col))
    return ast_util.enumerate_scope(rootpath)


class TestDerefAutoComplete(unittest.TestCase):
  def testDirectAutocomplete(self):
    suggestions = readAndAutocomplete("""
    bar = { y = 3 };
    x = bar.|
    """)
    self.assertSetEqual(set(['y']), set(suggestions))

  def testHalfAutocomplete(self):
    suggestions = readAndAutocomplete("""
    bar = { y = 3 };
    x = bar.y|
    """)
    self.assertSetEqual(set(['y']), set(suggestions))

  def testDoubleDerefAutocomplete(self):
    suggestions = readAndAutocomplete("""
    bar = { y = { z = 1 } };
    x = bar.y.|
    """)
    self.assertSetEqual(set(['z']), set(suggestions))

  def testParsedLocationOfIncompleteDoubleDeref(self):
    x = ast.lenient_grammar().expression.parseString('bar.y.')[0]
    left = x._haystack
    self.assertLess(left.location.end_offset, x.location.end_offset)


def readAndAutocomplete(source):
  source = source.strip()
  source, line, col = find_cursor(source)
  tree = gcl.reads(source, filename='input.gcl', allow_errors=True)
  q = gcl.SourceQuery('input.gcl', line, col - 1)
  ast_rootpath = tree.find_tokens(q)
  return ast_util.find_completions(ast_rootpath)


def find_cursor(source):
  """Return (source, line, col) based on the | character, stripping the source."""
  i = source.index('|')
  assert i != -1
  l = pyparsing.lineno(i, source)
  c = pyparsing.col(i, source)
  return source.replace('|', ''), l, c
