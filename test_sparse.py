import re
import sys
import unittest

from gcl import sparse
from gcl.sparse import T, ParseError


class TestCombinators(unittest.TestCase):
  def test_sequence(self):
    grammar = T('a') + T('b') + T('c')
    parse_string(grammar, 'abc')

    with self.assertRaises(ParseError):
      parse_string(grammar, 'ad')

  def test_either(self):
    grammar = T('a') | T('b') | T('c')
    parse_string(grammar, 'a')
    parse_string(grammar, 'b')
    parse_string(grammar, 'c')

    with self.assertRaises(ParseError):
      parse_string(grammar, 'd')

  def test_either_and_sequence(self):
    long = T('a') + T('b') + T('c')
    short = T('d')
    grammar = long | short

    parse_string(grammar, 'abc')
    parse_string(grammar, 'd')

    with self.assertRaises(ParseError):
      parse_string(grammar, 'ad')

  def test_zero_or_more(self):
    grammar = T('b') + sparse.ZeroOrMore(T('a')) + T('b')
    parse_string(grammar, 'bb')
    parse_string(grammar, 'bab')
    parse_string(grammar, 'baab')

    with self.assertRaises(ParseError):
      parse_string(grammar, 'baba')
      parse_string(grammar, 'bc')

  def test_incomplete_parse(self):
    grammar = T('a') + T('b') + T('c')

    with self.assertRaises(ParseError):
      parse_string(grammar, 'ab')

  def test_braced_list(self):
    grammar = sparse.braced_list(T('('), T('a'), T(','), T(')'))
    parse_string(grammar, '()')
    parse_string(grammar, '(a)')
    parse_string(grammar, '(a,)')
    parse_string(grammar, '(a,a)')
    parse_string(grammar, '(a,a,)')

    with self.assertRaises(ParseError):
      parse_string(grammar, '(aa)')
      parse_string(grammar, '(a,,)')

  def test_forward(self):
    # Test recursive grammars
    expr = sparse.Forward()
    nested = T('(') + expr + T(')')
    expr.set(nested | T('a'))

    parse_string(expr, 'a')
    parse_string(expr, '(a)')
    parse_string(expr, '((a))')

    with self.assertRaises(ParseError):
      parse_string(expr, '(a))')
      parse_string(expr, '((a)')

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

    parse_string(grammar, 'abcd')
    parse_string(grammar, 'abce')


class TestGrammarHelpers(unittest.TestCase):
  def test_parenthesized(self):
    grammar = sparse.parenthesized(T('a').action(lambda x: x))
    ret = parse_string(grammar, '(a)')
    self.assertEquals(['a'], ret)


class TestParseActionsAndCapture(unittest.TestCase):
  def test_capture_sequence_returns_result(self):
    grammar = T('a') + T('b') + T('c')
    g = grammar.action(lambda a, b, c: 'toet')
    ret = parse_string(g, 'abc')
    self.assertEquals(['toet'], ret)

  def test_suppress_capture(self):
    grammar = T('a') + T('b').suppress() + T('c')
    g = grammar.action(lambda a, c: 'toet')
    ret = parse_string(g, 'abc')
    self.assertEquals(['toet'], ret)

  def test_capture_distinguishes_either(self):
    grammar = ((T('a') + T('b')).action(lambda a, b: 'one')
              | T('c').action(lambda c: 'two'))

    self.assertEquals(['one'], parse_string(grammar, 'ab'))
    self.assertEquals(['two'], parse_string(grammar, 'c'))

  def test_optional_default_capture(self):
    grammar = sparse.Optional(T('a'), default='b') + T('c', capture=False)
    sparse.print_parser(grammar.make_parser(), sys.stdout)
    ret = parse_string(grammar, 'c')
    self.assertEquals(['b'], ret)

  def test_two_optionals(self):
    ab = sparse.Optional(T('a') + T('b'))
    c = sparse.Optional(T('c'))
    grammar = ab + c + T('d')

    parse_string(grammar, 'abcd')
    parse_string(grammar, 'abd')
    parse_string(grammar, 'cd')
    parse_string(grammar, 'd')

    with self.assertRaises(sparse.ParseError):
      parse_string(grammar, 'abcd')
      parse_string(grammar, 'ad')

  def test_alts_with_optional(self):
    grammar = (sparse.Optional(T('a')) | sparse.Optional(T('b'))) + T('c')

    parse_string(grammar, 'ac')
    parse_string(grammar, 'bc')
    parse_string(grammar, 'c')

  def test_error_reporting_wo_backtracking(self):
    grammar = sparse.Optional(T('a') - T('b') + T('c')) + T(';')

    try:
      parse_string(grammar, 'ab;')  # The error should tell me I forgot the c
      self.fail('Should have thrown')
    except sparse.ParseError as e:
      print e
      self.assertTrue('\'c\'' in str(e))

  def test_more_complex_capture(self):
    def grab(*args):
      return Box(args)

    cccab = (sparse.ZeroOrMore(T('c')) + T('a').action(grab)).action(grab)

    parse_string(cccab, 'a')

  def test_optional_after_no_backtracking(self):
    grammar = T('a') - sparse.Optional(T('b')) - T('c')

    parse_string(grammar, 'abc')
    parse_string(grammar, 'ac')

  def test_zero_or_more_with_nobacktracking_contents(self):
    grammar = T('a') + sparse.ZeroOrMore(T('&') - T('b')) + T('c')

    parse_string(grammar, 'a&bc')
    parse_string(grammar, 'a&b&bc')
    parse_string(grammar, 'ac')

    with self.assertRaises(sparse.ParseError):
      parse_string(grammar, 'a&c')
      parse_string(grammar, 'a&b&c')

  def test_nobacktracking_plus_zeroormore_breaks_backtracking(self):
    # FAILS because the - fucks up the backtracking of 'grammar's Alternative
    inner = T('a') - sparse.ZeroOrMore(T('a'))
    grammar = (inner + T(',') + inner) | (inner + T('?'))

    sparse.Trace.enable(True)

    with self.assertRaises(sparse.ParseError):
      parse_string(grammar, 'a?')

    # Introduce a Rule to make this work
    inner = sparse.Rule() >> T('a') - sparse.ZeroOrMore(T('a'))
    grammar = (inner + T(',') + inner) | (inner + T('?'))
    parse_string(grammar, 'a?')

  def test_report_failure_in_wrong_place(self):
    """Problem I actually ran into with optionals, where the error is reported in the wrong place."""
    num        = sparse.Rule() >> T('1') | T('2')
    expression = sparse.Rule() >> num - sparse.ZeroOrMore(T('+') - num)
    schema     = sparse.Rule() >> T(':') - T('y')
    value      = sparse.Rule() >> T('=') - expression
    rule       = sparse.Rule() >> T('x') - sparse.Optional(schema) + sparse.Optional(value, 5)

    try:
      parse_string(rule, 'x=1+3')
    except sparse.ParseError as e:
      print(str(e))
      assert "Unexpected '3'" in str(e)

  def test_rule_with_action(self):
    grammar = sparse.Rule() >> T('a') >> (lambda x: 'hi')
    out = parse_string(grammar, 'a')
    self.assertEquals(['hi'], out)


  # def test_error_reporting_in_default_optional(self):
    # inner = sparse.Optional(T('c') | sparse.ZeroOrMore(T('?')) + (T('a') | T('b')), default='')
    # grammar = T('{') - sparse.delimited_list(inner, T(',')) + T('}')

    # try:
      # parse_string(grammar, '{c,x}')
      # self.fail('Should have thrown')
    # except ParseError as e:
      # print e
      # self.fail('boom')


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


def parse_string(grammar, tokens):
  try:
    return sparse.parse_all(sparse.make_parser(grammar), ts(tokens))
  except ParseError as e:
    #sparse.print_parser(sparse.make_parser(grammar), sys.stdout)
    raise e.add_context('input.gcl', ''.join(str(t) for t in tokens))
