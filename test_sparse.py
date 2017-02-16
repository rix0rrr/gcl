import sys
import unittest

from gcl import sparse
from gcl.sparse import T, ParseError


class TestCombinators(unittest.TestCase):
  def test_sequence(self):
    grammar = T('a') + T('b') + T('c')
    grammar_parses(grammar, 'abc')

    with self.assertRaises(ParseError):
      grammar_fails(grammar, 'ad')

  def test_either(self):
    grammar = T('a') | T('b') | T('c')
    grammar_parses(grammar, 'a')
    grammar_parses(grammar, 'b')
    grammar_parses(grammar, 'c')

    with self.assertRaises(ParseError):
      grammar_fails(grammar, 'd')

  def test_either_and_sequence(self):
    long = T('a') + T('b') + T('c')
    short = T('d')
    grammar = long | short

    grammar_parses(grammar, 'abc')
    grammar_parses(grammar, 'd')

    with self.assertRaises(ParseError):
      grammar_fails(grammar, 'ad')

  def test_zero_or_more(self):
    grammar = T('b') + sparse.ZeroOrMore(T('a')) + T('b')
    grammar_parses(grammar, 'bb')
    grammar_parses(grammar, 'bab')
    grammar_parses(grammar, 'baab')

    with self.assertRaises(ParseError):
      grammar_fails(grammar, 'baba')
      grammar_fails(grammar, 'bc')

  def test_delimited_list(self):
    grammar = sparse.delimited_list(T('('), T('a'), T(','), T(')'))
    grammar_parses(grammar, '()')
    grammar_parses(grammar, '(a)')
    grammar_parses(grammar, '(a,)')
    grammar_parses(grammar, '(a,a)')
    grammar_parses(grammar, '(a,a,)')

    with self.assertRaises(ParseError):
      grammar_fails(grammar, '(aa)')
      grammar_fails(grammar, '(a,,)')

  def test_forward(self):
    # Test recursive grammars
    expr = sparse.Forward()
    nested = T('(') + expr + T(')')
    expr.set(nested | T('a'))

    grammar_parses(expr, 'a')
    grammar_parses(expr, '(a)')
    grammar_parses(expr, '((a))')

    with self.assertRaises(ParseError):
      grammar_fails(expr, '(a))')
      grammar_fails(expr, '((a)')

  def test_recognize_subroutines(self):
    """To keep the size of FSMs manageable.

    The grammar will automatically detect shared combinator sections,
    and turn those into subroutines that use pushed/popped states.
    """
    shared = T('a') + T('b') + T('c')
    exp1 = shared + T('d')
    exp2 = shared + T('e')
    grammar = exp1 | exp2

    # FIXME: Don't know yet how to test this properly...

    #grammar_parses(grammar, 'abcd')
    grammar_parses(grammar, 'abce')


class TestGrammarHelpers(unittest.TestCase):
  def test_parenthesized(self):
    grammar = sparse.parenthesized(T('a').action(lambda x: x.type))
    ret = grammar_parses(grammar, '(a)')
    self.assertEquals(('a',), ret)


class TestParseActionsAndCapture(unittest.TestCase):
  def test_capture_sequence_returns_result(self):
    grammar = T('a') + T('b') + T('c')
    g = grammar.action(lambda a, b, c: 'toet')
    debug_grammar(g)
    ret = grammar_parses(g, 'abc')
    self.assertEquals(('toet',), ret)

  def test_suppress_capture(self):
    grammar = T('a') + T('b').suppress() + T('c')
    g = grammar.action(lambda a, c: 'toet')
    ret = grammar_parses(g, 'abc')
    self.assertEquals(('toet',), ret)

  def test_capture_distinguishes_either(self):
    grammar = ((T('a') + T('b')).action(lambda a, b: 'one')
              | T('c').action(lambda c: 'two'))

    self.assertEquals(('one',), grammar_parses(grammar, 'ab'))
    self.assertEquals(('two',), grammar_parses(grammar, 'c'))

  def test_optional_default_capture(self):
    grammar = sparse.Optional(T('a'), default='b') + T('c', capture=False)
    ret = grammar_parses(grammar, 'c')
    self.assertEquals(('b',), ret)


class TestFSMSimplification(unittest.TestCase):
  def test_simplify_capture_plus_dispatch(self):
    end_state = sparse.State()
    fsm = sparse.State([sparse.Transition('a',
        target_state=sparse.State([sparse.Transition(sparse.EPSILON, dispatch_captures=object(),
          target_state=end_state)]))])
    simplified = sparse.simplify_fsm(fsm)

    # This should end up with 2 transitions simplified into 1
    self.assertEquals(end_state, simplified.transitions()[0].target_state)

  def test_simplify_sequence_of_three(self):
    end_state = sparse.State()
    fsm = sparse.State([sparse.Transition('a',
        target_state=sparse.State([sparse.Transition('b',
            target_state=sparse.State([sparse.Transition(sparse.EPSILON, dispatch_captures=object(),
              target_state=end_state)]))]))])
    simplified = sparse.simplify_fsm(fsm)

    sparse.graphvizify(sparse.ParserFSM(simplified, end_state), sys.stdout)

    # This should end up with 3 transitions simplified into 2
    self.assertEquals(end_state, simplified.transitions()[0].target_state.transitions()[0].target_state)

class TestTokenizer(unittest.TestCase):
  def test_tokens_are_simplified_correctly(self):
    scanner = sparse.Scanner([
      sparse.DOUBLE_QUOTED_STRING
      ])

    tokens = list(scanner.tokenize(r'"Can we \" scan it"'))
    self.assertEquals('dq_string', tokens[0].type)
    self.assertEquals('Can we \" scan it', tokens[0].value)

  def test_parse_integers(self):
    scanner = sparse.Scanner([
      sparse.WHITESPACE,
      sparse.INTEGER
      ])

    tokens = list(scanner.tokenize('1 2 4 8 15 16 23'))
    numbers = [t.value for t in tokens]
    self.assertEquals([1, 2, 4, 8, 15, 16, 23], numbers)


def ts(str):
  ret = []
  for i, t in enumerate(str):
    ret.append(sparse.Token(t, t, (i, i+1)))
  return ret


def grammar_fails(grammar, tokens):
  parser = sparse.make_parser(grammar)
  tokens = ts(tokens)
  parser.feed_all(tokens)
  parser.assert_finished()


def grammar_parses(grammar, tokens):
  parser = sparse.make_parser(grammar)
  tokens = ts(tokens)
  try:
    for i, token in enumerate(tokens):
      parser.feed(token)
    return parser.parsed_value()
  except Exception:
    print [t.type for t in tokens[:i+1]]
    sparse.graphvizify(parser, sys.stdout)
    raise


def debug_grammar(grammar):
  sparse.graphvizify(sparse.make_parser(grammar), sys.stdout)
