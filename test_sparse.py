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

  def test_oneormore_and_terminator(self):
    # sparse.delimited_list(expr, sparse.Q(t_sep))
    grammar = T('a') - sparse.OneOrMore(T('b')) - T(';')

    parse_string(grammar, 'ab;')

  def test_listmember_and_alternatives_notmatching(self):
    alts = (T('a') - T('=') - T('1')
        | T('b') - T('=') - T('2'))
    grammar = T('{') - sparse.delimited_list(alts, T(';')) - T('}')

    with self.assertRaises(sparse.ParseError):
      parse_string(grammar, '{nope}')

  def test_anything_except(self):
    grammar = T('a') + sparse.AnythingExcept('b', 'c') + T('d')

    parse_string(grammar, 'aad')

    with self.assertRaises(sparse.ParseError):
      parse_string(grammar, 'abd')

    with self.assertRaises(sparse.ParseError):
      parse_string(grammar, 'acd')

  def test_eat_failures(self):
    consume_parse_failures = sparse.ZeroOrMore(sparse.AnythingExcept(';', '}'))

    grammar = sparse.RecoverFailure(sparse.OneOrMore(T('a')), consume_parse_failures)

    parse_string(grammar, 'aaa')
    parse_string(grammar, 'bcd')

    with self.assertRaises(sparse.ParseError):
      # Don't know if I want this, but this is the behavior we have right now
      parse_string(grammar, 'aab')

  def test_never(self):
    grammar = T('a') + sparse.Never()

    with self.assertRaises(sparse.ParseError):
      parse_string(grammar, 'abd')


class TestGrammarHelpers(unittest.TestCase):
  def test_parenthesized(self):
    grammar = sparse.parenthesized(T('a').action(lambda _, x: x))
    ret = parse_string(grammar, '(a)')
    self.assertEquals(['a'], values(ret))


class TestParseActionsAndCapture(unittest.TestCase):
  def test_capture_sequence_returns_result(self):
    grammar = T('a') + T('b') + T('c')
    g = grammar.action(lambda _, a, b, c: 'toet')
    ret = parse_string(g, 'abc')
    self.assertEquals(['toet'], ret)

  def test_suppress_capture(self):
    grammar = T('a') + T('b').suppress() + T('c')
    g = grammar.action(lambda _, a, c: 'toet')
    ret = parse_string(g, 'abc')
    self.assertEquals(['toet'], ret)

  def test_capture_distinguishes_either(self):
    grammar = ((T('a') + T('b')).action(lambda _, a, b: 'one')
              | T('c').action(lambda _, c: 'two'))

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
      print(e)
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

  def test_optional_inside_backtracking(self):
    grammar = T('a') - sparse.Optional(T('b') - T('c'))

    parse_string(grammar, 'abc')
    with self.assertRaises(sparse.ParseError):
      parse_string(grammar, 'abd')

  def test_zero_or_more_with_nobacktracking_contents(self):
    grammar = T('a') + sparse.ZeroOrMore(T('&') - T('b')) + T('c')

    parse_string(grammar, 'a&bc')
    parse_string(grammar, 'a&b&bc')
    parse_string(grammar, 'ac')

    with self.assertRaises(sparse.ParseError):
      parse_string(grammar, 'a&c')
      parse_string(grammar, 'a&b&c')

  def test_nobacktracking_plus_zeroormore_breaks_backtracking(self):
    inner = T('a') - sparse.ZeroOrMore(T('a'))

    grammar = (inner + T(',') + inner) | (inner + T('?'))

    sparse.print_parser(grammar.make_parser(), sys.stdout)

    with self.assertRaises(sparse.ParseError):
      parse_string(grammar, 'a?')

  def test_zeroormore_within_alternatives(self):
    inner = T('a') + sparse.ZeroOrMore(T('a'))
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

  def test_report_longest_failure(self):
    grammar = (T('a') + T('b') + T('c') + T('d')
            | T('a') + T('d'))

    try:
      parse_string(grammar, 'abd')
    except sparse.ParseError as e:
      assert "expected 'c'" in str(e)

  def test_rule_with_action(self):
    grammar = sparse.Rule() >> T('a') >> (lambda _, x: 'hi')
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

    tokens = tokenize(scanner, r'"Can we \" scan it"')
    self.assertEquals('dq_string', tokens[0].type)
    self.assertEquals('Can we \" scan it', tokens[0].value)

  def test_parse_integers(self):
    scanner = sparse.Scanner([
      sparse.WHITESPACE,
      sparse.INTEGER
      ])

    tokens = tokenize(scanner, '1 2 4 8 15 16 23')
    numbers = [t.value for t in tokens if t.type != sparse.END_OF_FILE]
    self.assertEquals([1, 2, 4, 8, 15, 16, 23], numbers)


def tokenize(scanner, s):
  f = sparse.File('<input>', s)
  return list(scanner.tokenize(f))



class TestErrorReporting(unittest.TestCase):
  def test_find_line(self):
    self.do_index_test_two_ways(1, 'ab', 0, '|ab\ncd\nef')
    self.do_index_test_two_ways(1, 'ab', 2, 'ab|\ncd\nef')
    self.do_index_test_two_ways(2, 'cd', 0, 'ab\n|cd\nef')
    self.do_index_test_two_ways(2, 'cd', 1, 'ab\nc|d\nef')
    self.do_index_test_two_ways(2, 'cd', 2, 'ab\ncd|\nef')
    self.do_index_test_two_ways(3, 'ef', 2, 'ab\ncd\nef|')

  def do_index_test_two_ways(self, expected_line_nr, expected_text, expected_col, source_with_marker):
    plain_text, marker_ix = line_with_marker(source_with_marker)

    line_nr, text, col = sparse.find_line_context(plain_text, marker_ix)
    self.assertEquals((expected_line_nr, expected_text, expected_col), (line_nr, text, col))

    found_ix = sparse.linecol_to_index(plain_text, line_nr, col)
    self.assertEquals(marker_ix, found_ix)


def line_with_marker(text):
  i = text.index('|')
  return text[:i] + text[i+1:], i


class Box(object):
  def __init__(self, thing):
    self.thing = thing


def ts(file):
  scanner = sparse.Scanner([
      sparse.Syntax('character', '.'),
      ])
  # Use the value as type for every token
  for t in scanner.tokenize(file):
    if t.type == sparse.END_OF_FILE:
      yield t
    else:
      yield sparse.Token(t.value, t.value, t.span)


def parse_string(grammar, tokens):
  file = sparse.File('<input>', tokens)
  return sparse.parse_all(sparse.make_parser(grammar), ts(file), file)


def values(tokens):
  return [t.value for t in tokens]
