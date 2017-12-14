import unittest

import gcl
from gcl import ast_util
from gcl import ast2 as ast
from gcl import sparse
from gcl import framework

class TestEnumerateScope(unittest.TestCase):
  def testSiblingsInScope(self):
    scope = readAndQueryScope('|henk = 5')
    assert 'henk' in scope

  def testScopeObjectHasLocation(self):
    scope = readAndQueryScope('|henk = 5')
    self.assertEquals(1, scope['henk'].span.line_nr)
    self.assertEquals(0, scope['henk'].span.col_nr)

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
    assert scope['henk'].span.line_nr == 2  # Find the right henk

  def testIncludeDefaultBuiltins(self):
    source = 'henk = 5|'
    source, line, col = find_cursor(source)
    tree = gcl.reads(source, filename='input.gcl')
    rootpath = tree.find_tokens(gcl.SourceQuery('input.gcl', line, col))
    scope = ast_util.enumerate_scope(rootpath, include_default_builtins=True)
    print scope.keys()
    assert 'len' in scope


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

  def testParseErrorDontMakeTuplesUnbalanced(self):
    # Original error is that parse recovery here will at until the
    # next "synchronizing" }, which will actually desynchronize the parser
    # completely.
    suggestions = readAndQueryScope("""
      bier|
      queues = {}
    """, allow_errors=True)
    # It's already cool if we don't fail parsing this


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

  # def testParsedLocationOfIncompleteDoubleDeref(self):
    # x = ast.make_grammar(allow_errors=True).expression.parseString('bar.y.')[0]
    # left = x._haystack
    # self.assertLess(left.span.end, x.span.end)

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

  def testAutoCompleteFromBaseTupleInIdentifierPosition(self):
    suggestions = readAndAutocomplete("""
    base = {
      key;
    };
    bar = base {
      k|
    };
    """)
    self.assertSetEqual(set(['key']), set(suggestions))

  def testAutoCompleteFromBaseTupleInIdentifierPositionNoPrefix(self):
    suggestions = readAndAutocomplete("""
    base = {
      key;
    };
    bar = base {
      |
    };
    """)
    self.assertSetEqual(set(['key']), set(suggestions))

  def testAutoCompleteFromBaseTupleInIdentifierPositionInList(self):
    suggestions = readAndAutocomplete("""
    base = {
      key;
    };
    bar = [ base {
      |
    }];
    """)
    self.assertSetEqual(set(['key']), set(suggestions))

  def testAutoCompleteBetweenKeys(self):
    suggestions = readAndAutocomplete("""
    base = {
      key;
    };
    bar = base {
      x = 3;
      |
      y = 4;
    };
    """)
    self.assertSetEqual(set(['key']), set(suggestions))

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

  def testAutoCompleteWithListComprehension(self):
    suggestions = readAndAutocomplete("""
    bla = { values = [1, 2, 3, 4] };
    y = [x * 2 for x in bla.v| ];
    """)
    self.assertSetEqual(set(['values']), set(suggestions))

  def testAutoCompleteWithUnboundValue(self):
    suggestions = readAndAutocomplete("""
    bla;
    y = bla.|;
    """)
    self.assertSetEqual(set([]), set(suggestions))

  def testAutoCompleteBrokenIndexSyntax(self):
    suggestions = readAndAutocomplete("""
    list = [{ x = 1 }];
    value = list[0].|
    """)
    self.assertSetEqual(set(['list', 'value']), set(suggestions))

  def testAutoCompleteInList(self):
    suggestions = readAndAutocomplete("""
    list = [{ x = 1 }];
    value = list(0).|
    """)
    self.assertSetEqual(set(['x']), set(suggestions))

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

  def testHideSchemasFromEnumeration(self):
    suggestions = readAndAutocomplete("""
      x = fo|
    """, root_env=gcl.default_env)
    self.assertNotIn('string', suggestions)



class TestFindValue(unittest.TestCase):
  def testCantFindValueAtTopLevel(self):
    found = readAndFindValue("""
    tup|le = { value = 1 };
    """)
    self.assertEquals(None, found)

  def testDoFindSideValues(self):
    found = readAndFindValue("""
    something = 3;
    other = some|thing;
    """)
    self.assertEquals(3, found)

  def testInherit(self):
    found = readAndFindValue("""
    something = 3;
    other = { inherit somet|hing }
    """)
    self.assertEquals(3, found)

  def testHoverOnInclude(self):
    found = readAndFindValue("""
    y = 1;
    something = include 'be|rt' { inherit y; x = 3; }
    """)
    self.assertEquals('bert', found)


def readAndQueryScope(source, **kwargs):
  source, line, col = find_cursor(source)
  tree = gcl.reads(source, filename='input.gcl', **kwargs)
  rootpath = tree.find_tokens(gcl.SourceQuery('input.gcl', line, col))
  return ast_util.enumerate_scope(rootpath)


def readAndAutocomplete(source, root_env=gcl.Environment({})):
  source, line, col = find_cursor(source)
  tree = gcl.reads(source, filename='input.gcl', allow_errors=True)
  return ast_util.find_completions_at_cursor(tree, 'input.gcl', line, col, root_env=root_env)


def readAndFindValue(source):
  source, line, col = find_cursor(source)
  tree = gcl.reads(source, filename='input.gcl', allow_errors=True)
  return ast_util.find_value_at_cursor(tree, 'input.gcl', line, col)


def find_cursor(source):
  """Return (source, line, col) based on the | character, stripping the source."""
  source = source.strip()
  i = source.index('|')
  assert i != -1
  span = sparse.Span(i, i, sparse.File('', source))
  line, _, col = span.line_context()

  source = source.replace('|', '')

  sq = sparse.SourceQuery('input.gcl', line, col+1)
  span2 = sparse.query_to_span(sq, sparse.File('input.gcl', source))

  return source, line, col + 1
