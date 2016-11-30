import unittest

import pyparsing

import gcl
from gcl import ast_util
from gcl import ast
from gcl import framework

class TestEnumerateScope(unittest.TestCase):
  def testSiblingsInScope(self):
    scope = readAndQueryScope('|henk = 5')
    assert 'henk' in scope

  def testScopeObjectHasLocation(self):
    scope = readAndQueryScope('|henk = 5')
    self.assertEquals(1, scope['henk'].location.lineno)
    self.assertEquals(1, scope['henk'].location.col)

  def testInherited(self):
    source = """
    henk = 5;
    other = {
      f|oo = 1;
    }
    """
    scope = readAndQueryScope(source)
    assert 'henk' in scope
    assert 'foo' in scope

  def testCompositeNeg(self):
    source = """
    A = { henk = 5 };
    B = A { fo|o = 1 };
    """
    scope = readAndQueryScope(source)  # foo
    assert 'henk' not in scope
    assert 'foo' in scope

  def testCompositeMixin(self):
    source = """
    A = { henk = 5 };
    B = A { henk; foo| = 1 };
    """
    scope = readAndQueryScope(source)  # foo
    assert 'henk' in scope
    assert 'foo' in scope

  def testTupleWithInherits(self):
    source = """
    henk = 5;
    B = { inherit henk| }
    """
    scope = readAndQueryScope(source)  # inner tuple
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
      brok|en = + + ax 89sdf/b8;
      post_inner = 3;
    }
    """, allow_errors=True)
    self.assertSetEqual(set(['outer', 'tup', 'pre_inner', 'broken', 'post_inner']), set(scope.keys()))

  def testUnparseableTupleLineMissingSemicolon(self):
    scope = readAndQueryScope("""
    outer = 1;
    tup = {
      pre_inner = 2;
      brok|en = + + ax 89sdf/b8
      post_inner = 3;
    }
    """, allow_errors=True)
    self.assertSetEqual(set(['outer', 'tup', 'pre_inner', 'broken']), set(scope.keys()))

  def testUnparseableTupleLineCompleteGarbage(self):
    scope = readAndQueryScope("""
    outer = 1;
    tup = {
      pre_inner = 2;
      + +| ax 89sdf/b8;
      post_inner = 3;
    }
    """, allow_errors=True)
    self.assertSetEqual(set(['outer', 'tup', 'pre_inner', 'post_inner']), set(scope.keys()))

  def testMissingEqualsRecover(self):
    scope = readAndQueryScope("""
      foo = 3;

      bar = {
          bip|pety bop;
      };
      """, allow_errors=True)
    self.assertSetEqual(set(['foo', 'bar', 'bippety']), set(scope.keys()))

  def testTrailingWordRecover(self):
    scope = readAndQueryScope("""
      foo = 3;
      |f
      """, allow_errors=True)
    self.assertSetEqual(set(['foo', 'f']), set(scope.keys()))

  def testMissingClosingBraceRecover(self):
    scope = readAndQueryScope("""
    outer = 1;
    tup = {
      inn|er = 2;
    """, allow_errors=True)
    self.assertSetEqual(set(['outer', 'inner', 'tup']), set(scope.keys()))

  def testMissingClosingBraceRecoverDouble(self):
    scope = readAndQueryScope("""
    outer = 1;
    tup = {
      tap = {
        inn|er = 2;
    };
    """, allow_errors=True)
    self.assertSetEqual(set(['outer', 'inner', 'tup', 'tap']), set(scope.keys()))

  def testRecoverDoubleDefinition(self):
    scope = readAndQueryScope("""
    foo = 1;
    |foo = 2;
    """, allow_errors=True)
    self.assertSetEqual(set(['foo']), set(scope.keys()))

  def testRecoverIncompleteDeref(self):
    scope = readAndQueryScope("""
    |foo = bar.
    """, allow_errors=True)
    self.assertSetEqual(set(['foo']), set(scope.keys()))

  def testRecoverDoubleEquals(self):
    suggestions = readAndQueryScope("""
    x = |
    y = 3;
    """, allow_errors=True)
    self.assertSetEqual(set(['x']), set(suggestions))

  def testRecoverBinOp(self):
    suggestions = readAndQueryScope("""
    y = 1 + 2 + |;
    """, allow_errors=True)
    self.assertSetEqual(set(['y']), set(suggestions))



class TestAutoComplete(unittest.TestCase):
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

  def testNonDerefAutocomplete(self):
    suggestions = readAndAutocomplete("""
    bar = { y = { z = 1 } };
    x = ba|
    """)
    self.assertSetEqual(set(['bar', 'x']), set(suggestions))

  def testNoAutocompleteInIdentifierPosition(self):
    suggestions = readAndAutocomplete("""
    bar = {
      henk = 3;
      le|
    };
    """)
    self.assertSetEqual(set([]), set(suggestions))

  def testAutoCompleteDocsScope(self):
    suggestions = readAndAutocomplete("""
    #. this is a foo
    #.
    #. there are many like it but this one is mine
    foo = { x = 3; };
    bar = f|
    """)
    self.assertEquals('this is a foo\n\nthere are many like it but this one is mine', suggestions['foo'].doc)

  def testAutoCompleteDocsBuiltin(self):
    suggestions = readAndAutocomplete("""
    bar = compo|
    """, root_env=gcl.default_env)
    print(suggestions)
    self.assertTrue('Compose all given tuples' in suggestions['compose_all'].doc)

  def testAutoCompleteDocsDeref(self):
    suggestions = readAndAutocomplete("""
    foo = {
      #. yes
      really;
    };
    bar = foo.reall|
    """, root_env=gcl.default_env)
    self.assertEquals('yes', suggestions['really'].doc)

  def testCompleteDoubleEquals(self):
    suggestions = readAndAutocomplete("""
    x = |
    y = 3;
    """)
    self.assertSetEqual(set(['x']), set(suggestions))

  def testCompleteBinOp(self):
    suggestions = readAndAutocomplete("""
    y = 1 + 2 + |;
    """)
    self.assertSetEqual(set(['y']), set(suggestions))

  def testCompletePastIncludesWhenFileChangesAndCachingDisabled(self):
    includable = ["before = 1;"]
    def load_from_var(base, rel, env=None):
      return gcl.loads(includable[0], env=env)

    source = """
    inc = include 'inc.gcl';
    bar = inc.|
    """.strip()
    source, line, col = find_cursor(source)
    tree = gcl.reads(source, filename='input.gcl', allow_errors=True, loader=load_from_var)

    with framework.DisableCaching():
      completions = ast_util.find_completions_at_cursor(tree, 'input.gcl', line, col)
    self.assertTrue('before' in completions)

    includable[0] = "after = 2;"

    with framework.DisableCaching():
      completions = ast_util.find_completions_at_cursor(tree, 'input.gcl', line, col)
    self.assertTrue('after' in completions)


def readAndQueryScope(source, **kwargs):
    source, line, col = find_cursor(source)
    tree = gcl.reads(source, filename='input.gcl', **kwargs)
    rootpath = tree.find_tokens(gcl.SourceQuery('input.gcl', line, col))
    return ast_util.enumerate_scope(rootpath)


def readAndAutocomplete(source, root_env=None):
  source, line, col = find_cursor(source)
  tree = gcl.reads(source, filename='input.gcl', allow_errors=True)
  return ast_util.find_completions_at_cursor(tree, 'input.gcl', line, col, root_env=root_env)


def find_cursor(source):
  """Return (source, line, col) based on the | character, stripping the source."""
  source = source.strip()
  i = source.index('|')
  assert i != -1
  l = pyparsing.lineno(i, source)
  c = pyparsing.col(i, source)
  return source.replace('|', ''), l, c
