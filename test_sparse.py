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

  def test_incomplete_parse(self):
    grammar = T('a') + T('b') + T('c')

    with self.assertRaises(ParseError):
      grammar_fails(grammar, 'ab')

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

  def test_simplify_sequences(self):
    expr = T('a') + T('b') + T('c')
    parser = expr.make_parser()
    self.assertEquals(3, len(parser.parsers))

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

    grammar_parses(grammar, 'abcd')
    grammar_parses(grammar, 'abce')


class TestGrammarHelpers(unittest.TestCase):
  def test_parenthesized(self):
    grammar = sparse.parenthesized(T('a').action(lambda x: x))
    ret = grammar_parses(grammar, '(a)')
    self.assertEquals(['a'], ret)


class TestParseActionsAndCapture(unittest.TestCase):
  def test_capture_sequence_returns_result(self):
    grammar = T('a') + T('b') + T('c')
    g = grammar.action(lambda a, b, c: 'toet')
    ret = grammar_parses(g, 'abc')
    self.assertEquals(['toet'], ret)

  def test_suppress_capture(self):
    grammar = T('a') + T('b').suppress() + T('c')
    g = grammar.action(lambda a, c: 'toet')
    ret = grammar_parses(g, 'abc')
    self.assertEquals(['toet'], ret)

  def test_capture_distinguishes_either(self):
    grammar = ((T('a') + T('b')).action(lambda a, b: 'one')
              | T('c').action(lambda c: 'two'))

    self.assertEquals(['one'], grammar_parses(grammar, 'ab'))
    self.assertEquals(['two'], grammar_parses(grammar, 'c'))

  def test_optional_default_capture(self):
    grammar = sparse.Optional(T('a'), default='b') + T('c', capture=False)
    sparse.print_parser(grammar.make_parser(), sys.stdout)
    ret = grammar_parses(grammar, 'c')
    self.assertEquals(['b'], ret)

  def test_two_optionals(self):
    ab = sparse.Optional(T('a') + T('b'))
    c = sparse.Optional(T('c'))
    grammar = ab + c + T('d')

    grammar_parses(grammar, 'abcd')
    grammar_parses(grammar, 'abd')
    grammar_parses(grammar, 'cd')
    grammar_parses(grammar, 'd')

    with self.assertRaises(sparse.ParseError):
      grammar_fails(grammar, 'abcd')
      grammar_fails(grammar, 'ad')

  def test_alts_with_optional(self):
    grammar = (sparse.Optional(T('a')) | sparse.Optional(T('b'))) + T('c')

    grammar_parses(grammar, 'ac')
    grammar_parses(grammar, 'bc')
    grammar_parses(grammar, 'c')

  def test_error_reporting_wo_backtracking(self):
    grammar = sparse.Optional(T('a') - T('b') + T('c')) + T(';')

    try:
      grammar_fails(grammar, 'ab;')  # The error should tell me I forgot the c
      self.fail('Should have thrown')
    except sparse.ParseError as e:
      print e
      self.assertTrue('\'c\'' in str(e))

  def test_more_complex_capture(self):
    def grab(*args):
      return Box(args)

    cccab = (sparse.ZeroOrMore(T('c')) + T('a').action(grab)).action(grab)

    grammar_parses(cccab, 'a')

  def test_optional_after_no_backtracking(self):
    grammar = T('a') - sparse.Optional(T('b')) - T('c')

    grammar_parses(grammar, 'abc')
    grammar_parses(grammar, 'ac')

  def test_zero_or_more_with_nobacktracking_contents(self):
    grammar = T('a') + sparse.ZeroOrMore(T('&') - T('b')) + T('c')

    grammar_parses(grammar, 'a&bc')
    grammar_parses(grammar, 'a&b&bc')
    grammar_parses(grammar, 'ac')

    with self.assertRaises(sparse.ParseError):
      grammar_fails(grammar, 'a&c')
      grammar_fails(grammar, 'a&b&c')


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


class TestErrorReporting(unittest.TestCase):
  def test_find_line(self):
    self.assertEquals((0, 'ab', 0), sparse.find_line_context(*line_with_marker('|ab\ncd\nef')))
    self.assertEquals((0, 'ab', 2), sparse.find_line_context(*line_with_marker('ab|\ncd\nef')))
    self.assertEquals((1, 'cd', 0), sparse.find_line_context(*line_with_marker('ab\n|cd\nef')))
    self.assertEquals((1, 'cd', 1), sparse.find_line_context(*line_with_marker('ab\nc|d\nef')))
    self.assertEquals((1, 'cd', 2), sparse.find_line_context(*line_with_marker('ab\ncd|\nef')))
    self.assertEquals((2, 'ef', 2), sparse.find_line_context(*line_with_marker('ab\ncd\nef|')))


def line_with_marker(text):
  i = text.index('|')
  return text[:i] + text[i+1:], i


class Box(object):
  def __init__(self, thing):
    self.thing = thing


def ts(str):
  for i, t in enumerate(str):
   yield sparse.Token(t, t, (i, i+1))


def grammar_fails(grammar, tokens):
  sparse.parse_all(sparse.make_parser(grammar), ts(tokens))


def grammar_parses(grammar, tokens):
  try:
    return sparse.parse_all(sparse.make_parser(grammar), ts(tokens))
  except ParseError:
    sparse.print_parser(sparse.make_parser(grammar), sys.stdout)
    raise
