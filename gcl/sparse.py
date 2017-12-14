# coding=utf-8
"""sparse -- A faster parser combinator library for Python.

Written when we outgrew the performance of pyparsing:

  - Uses a tokenizer: parses tokens instead of characters.
  - Does not use exceptions to report parse failures.
  - Uses jump tables to speed up parsing.

The tokenizer is based on regex module tricks from:

    http://lucumr.pocoo.org/2015/11/18/pythons-hidden-re-gems/
"""
from __future__ import unicode_literals

import collections
import copy
import itertools
import functools
import copy
import re
import sys

from sre_parse import Pattern, SubPattern, parse
from sre_compile import compile as sre_compile
from sre_constants import BRANCH, SUBPATTERN


__all__ = [
  'And',
  'AnythingExcept',
  'RecoverFailure',
  'DOUBLE_QUOTED_STRING',
  'END_OF_FILE',
  'FLOAT',
  'INTEGER',
  'NO_DEFAULT',
  'SINGLE_QUOTED_STRING',
  'WHITESPACE',
  'Either',
  'EmptySpan',
  'File',
  'Forward',
  'LazyList',
  'Never',
  'OneOrMore',
  'Optional',
  'ParseError',
  'ParseFailure',
  'ParseState',
  'ParseSuccess',
  'Q',
  'Rule',
  'Scanner',
  'SourceQuery',
  'Span',
  'Splat',
  'Symbol',
  'T',
  'Token',
  'TokenStream',
  'Trace',
  'ZeroOrMore',
  'braced_list',
  'delimited_list',
  'empty_span',
  'make_parser',
  'parenthesized',
  'parse',
  'parse_all',
  'print_parser',
  'query_to_span',
  'quoted_string_process',
  'quoted_string_regex',
  ]


class Symbol(object):
  def __init__(self, name):
    self.name = name

  def __repr__(self):
    return self.name


END_OF_FILE = Symbol('end-of-file')
AND_MORE = Symbol('and-more')
NO_DEFAULT = Symbol('NO_DEFAULT')


class ParseError(RuntimeError):
  def __init__(self, span, message):
    RuntimeError.__init__(self, message)
    assert isinstance(span, Span)
    self.span = span

  def context(self):
    return '\n'.join(self.span.annotated_source(self.message))


#--------------------------------------------------------------------------------
#  TOKENIZER


class File(object):
  def __init__(self, filename, contents):
    self.filename = filename
    self.contents = contents

  def __repr__(self):
    return 'File(%r)' % self.filename


class Span(collections.namedtuple('Span', ('begin', 'end', 'file'))):
  def contains_span(self, rhs):
    """Return True if rhs is fully within lhs."""
    return ((self.file == rhs.file)
        and (self.begin <= rhs.begin) and (rhs.end <= self.end))

  def contains_query(self, q):
    if q.filename != self.file.filename:
      return False

    span = query_to_span(q, self.file)
    return self.begin <= span.begin and span.end <= self.end

  def line_context(self):
    """Return the line number and complete line of the current span in the original file.

    Returns:
      Tuple of (line_nr, line_text, offset_in_line). line_nr is 1-based. Offset-in-line is 0-based.
    """
    return find_line_context(self.file.contents, self.begin)

  def original_string(self):
    _, line_text, offset_in_line = self.line_context()
    return line_text[offset_in_line:offset_in_line + self.end - self.begin]

  def annotated_source(self, message):
    """Returns a list of two strings, one with an error message and one with a position indicator right under it."""
    line_nr, line_text, offset_in_line = self.line_context()

    file_indicator = '%s:%d: ' % (self.file.filename, line_nr)
    return [file_indicator + line_text,
            ' ' * (len(file_indicator) + offset_in_line) + ('^' * min(8, max(1, len(self)))) + ' ' + message]

  @property
  def line_nr(self):
    """Return the 1-based line nr."""
    line_nr, _, _ = self.line_context()
    return line_nr

  @property
  def col_nr(self):
    """Return the 0-based column number."""
    _, _, col_nr = self.line_context()
    return col_nr

  def __add__(self, rhs):
    if rhs is empty_span:
      return self

    if self.file != rhs.file:
      raise ValueError('Cannot add two Spans from different files')

    return Span(min(self.begin, rhs.begin), max(self.end, rhs.end), self.file)

  def __len__(self):
    return self.end - self.begin

  def until(self, rhs):
    return Span(self.begin, rhs.end, self.file)

  __radd__ = __add__


class EmptySpan(object):
  def __init__(self):
    Span.__init__(self)

  def contains_span(self, rhs):
    return False

  def contains_query(self, rhs):
    return False

  def annotated_source(self, message):
    return ['<no file>:0: no source',
            '             ^^^^ ' + message]

  def __add__(self, rhs):
    return rhs

empty_span = EmptySpan()

# SourceQuery has 1-based line and column numbers.
SourceQuery = collections.namedtuple('SourceQuery', ['filename', 'line', 'col'])

QUERY_TO_SPAN_CACHE = {}

def query_to_span(query, file):
  if file.filename != query.filename:
    raise ValueError('Cannot convert query to span for nonmatching filename')

  key = (query, file)
  if key not in QUERY_TO_SPAN_CACHE:
    QUERY_TO_SPAN_CACHE.clear()

    ix = linecol_to_index(file.contents, query.line, query.col - 1)
    QUERY_TO_SPAN_CACHE[key] = Span(ix, ix, file)

  return QUERY_TO_SPAN_CACHE[key]


def all_newlines(s):
  """Generator that returns all newlines in the given string."""
  i = s.find('\n', 0)
  while i != -1:
    yield i
    i = s.find('\n', i)


class Token(object):
  __slots__ = ('type', 'value', 'span')

  def __init__(self, type, value, span):
    self.type = type
    self.value = value
    self.span = span

  def __repr__(self):
    return 'Token(%r, %r, %r)' % (self.type, self.value, self.span)


def quoted_string_regex(quote):
  return quote + '(?:[^' + quote + r'\\]|\\.)*' + quote


def quoted_string_process(value):
  return value[1:-1].decode('string_escape')


class Syntax(object):
  def __init__(self, token_type, regex, post_processor=None):
    self.token_type = token_type
    self.regex = regex
    self.post_processor = post_processor


WHITESPACE = Syntax('whitespace', r'\s+')
DOUBLE_QUOTED_STRING = Syntax('dq_string', quoted_string_regex('"'), quoted_string_process)
SINGLE_QUOTED_STRING = Syntax('sq_string', quoted_string_regex("'"), quoted_string_process)
FLOAT = Syntax('float', r'-?\d+\.\d+', float)
INTEGER = Syntax('integer', r'-?\d+', int)


class Scanner(object):
  def __init__(self, rules):
    pattern = Pattern()
    pattern.flags = re.M
    pattern.groups = len(rules) + 1

    # Validate the rules, if we do it later mistakes become hard to pinpoint
    for rule in rules:
      try:
        re.compile(rule.regex)
      except Exception:
        raise RuntimeError('Error parsing regex for token %s: %r' % (rule.token_type, rule.regex))

    self.rules = rules
    self._scanner = sre_compile(SubPattern(pattern, [
        (BRANCH, (None, [SubPattern(pattern, [
            (SUBPATTERN, (group, parse(r.regex, pattern.flags, pattern))),
        ]) for group, r in enumerate(rules, 1)]))
    ])).scanner

  def tokenize(self, file):
    assert isinstance(file, File)
    self.file = file
    for token, match, proc in self.scan(file.contents):
      if token == 'whitespace' or token == 'comment':
        continue
      if token == 'keyword' or token == 'symbol':
        token = match.group()
      value = proc(match.group()) if proc else match.group()
      begin, end = match.span()
      yield Token(token, value, Span(begin, end, file))
    yield Token(END_OF_FILE, 'end-of-file', Span(len(file.contents), len(file.contents), file))

  def scan(self, string, skip=False):
    sc = self._scanner(string)
    len_string = len(string)

    match = None
    for match in iter(sc.search if skip else sc.match, None):
      rule = self.rules[match.lastindex - 1]
      yield rule.token_type, match, rule.post_processor

    if match and match.end() < len_string:
      span = (match.end(), match.end() + 1)
      raise ParseError(Span(span[0], span[1], self.file), 'Unable to match token at %s' % (string[match.end():match.end() + 10] + '...'))


#--------------------------------------------------------------------------------
#  GRAMMAR

class Grammar(object):
  """A node in a Parser Combinator Grammar."""
  def __init__(self):
    self.refcount = 0
    self.name = None

  def action(self, action):
    return CapturingGrammar(self, action)

  def with_name(self, name):
    ret = copy.copy(self)
    ret.name = name
    return ret

  def take_reference(self):
    self.refcount += 1
    return self

  def __add__(self, rhs):
    return And(self, rhs)

  def __sub__(self, rhs):
    return self + NO_BACKTRACKING + rhs

  def __or__(self, rhs):
    return Either(self, rhs)


class StopBacktrackingGrammar(Grammar):
  def make_parser(self):
    return NO_BACKTRACKING_PARSER

NO_BACKTRACKING = StopBacktrackingGrammar()


class CapturingGrammar(Grammar):
  def __init__(self, inner, action):
    Grammar.__init__(self)
    self.inner = inner.take_reference()
    self.action = action

  def make_parser(self):
    return CapturingParser(self.inner.make_parser(), self.action, self.name)


class RecoverFailure(Grammar):
  def __init__(self, inner, resync):
    Grammar.__init__(self)
    self.inner = inner.take_reference()
    self.resync = resync.take_reference()

  def make_parser(self):
    return RecoverFailureParser(self.inner.make_parser(), self.resync.make_parser(), self.name)


class T(Grammar):
  def __init__(self, token_type, capture=True):
    Grammar.__init__(self)
    self.token_type = token_type
    self.capture = capture

  def suppress(self):
    return T(self.token_type, capture=False)

  def make_parser(self):
    return AtomParser(self.token_type, self.capture, self.name)


class Never(Grammar):
  def __init__(self):
    Grammar.__init__(self)

  def make_parser(self):
    return FailParser(self.name)


def Q(token_type):
  """Quiet token."""
  return T(token_type, capture=False)





class And(Grammar):
  def __init__(self, left, right):
    assert(isinstance(left, Grammar) and isinstance(right, Grammar))
    Grammar.__init__(self)
    self.left = left.take_reference()
    self.right = right.take_reference()

  def make_parser(self):
    left_parser = self.left.make_parser()
    right_parser = self.right.make_parser()

    return SequenceParser([left_parser, right_parser], self.name)


class AnythingExcept(Grammar):
  def __init__(self, *token_types):
    Grammar.__init__(self)
    self.token_types = token_types
    self.capture = True

  def suppress(self):
    ret = AnythingExcept(*self.token_types)
    ret.capture = False
    return ret

  def make_parser(self):
    return AnythingExceptParser(list(self.token_types) + [END_OF_FILE], self.capture, self.name)


class Never(Grammar):
  def __init__(self):
    Grammar.__init__(self)

  def make_parser(self):
    return FailParser(self.name)


def Q(token_type):
  """Quiet token."""
  return T(token_type, capture=False)





class And(Grammar):
  def __init__(self, left, right):
    assert(isinstance(left, Grammar) and isinstance(right, Grammar))
    Grammar.__init__(self)
    self.left = left.take_reference()
    self.right = right.take_reference()

  def make_parser(self):
    left_parser = self.left.make_parser()
    right_parser = self.right.make_parser()

    return SequenceParser([left_parser, right_parser], self.name)


class Optional(Grammar):
  def __init__(self, inner, default=NO_DEFAULT):
    Grammar.__init__(self)
    self.inner = inner.take_reference()
    self.default = default

  def make_parser(self):
    return OptionalParser(self.inner.make_parser(), self.name, missing_default=self.default)
    #return CountParser(self.inner.make_parser(), 0, 1, self.name, missing_default=self.default)


class Either(Grammar):
  def __init__(self, left, right):
    assert(isinstance(left, Grammar) and isinstance(right, Grammar))
    Grammar.__init__(self)
    self.left = left.take_reference()
    self.right = right.take_reference()

  def make_parser(self):
    left_parser = self.left.make_parser()
    right_parser = self.right.make_parser()

    return AlternativesParser([left_parser, right_parser], self.name)


class ZeroOrMore(Grammar):
  def __init__(self, inner):
    assert isinstance(inner, Grammar)
    Grammar.__init__(self)
    self.inner = inner.take_reference()

  def make_parser(self):
    return CountParser(self.inner.make_parser(), 0, None, self.name)


class OneOrMore(Grammar):
  def __init__(self, inner):
    assert isinstance(inner, Grammar)
    Grammar.__init__(self)
    self.inner = inner.take_reference()

  def make_parser(self):
    return CountParser(self.inner.make_parser(), 1, None, self.name)


def delimited_list(expr, sep):
  """Delimited list with trailing separator."""
  return Optional(expr + ZeroOrMore(sep - Optional(expr)))


def braced_list(opener, expr, sep, closer):
  """Delimited list with optional terminating separator."""
  return opener - delimited_list(expr, sep) - closer


def parenthesized(expr):
  """Parenthesized version of the inner expression."""
  return Q('(') - expr + Q(')')


class Forward(Grammar):
  def __init__(self):
    Grammar.__init__(self)
    self.inner = None
    self.parser = None

  def set(self, inner):
    assert isinstance(inner, Grammar)
    self.inner = inner.take_reference()

  def make_parser(self):
    if not self.inner:
      raise RuntimeError('Forward grammar never initialized')

    if not self.parser:
      self.parser = ForwardParser(self.name)
      self.parser.set(self.inner.make_parser())

    return self.parser


class Rule(Grammar):
  """Production Rule, this limits "stop backtracking" operators to the scope of this rule.

  We add an action to it because this is USUALLY what you want anyway.
  """

  def __init__(self, name=None, grammar=None, action=None):
    Grammar.__init__(self)
    self.name = name
    self.grammar = grammar.take_reference() if grammar else None
    self.action = action

  def __rshift__(self, other):
    if self.grammar is None:
      assert isinstance(other, Grammar)
      self.grammar = other.take_reference()
      return self
    elif self.action is None:
      assert callable(other)
      self.action = other
      return self
    else:
      raise RuntimeError('Too many >> arguments to a Rule')

  def make_parser(self):
    if self.action:
      return CapturingParser(self.grammar.make_parser(), self.action, self.name)
    else:
      return self.grammar.make_parser()


#--------------------------------------------------------------------------------
#  PARSERS

def trace_parser(method):
  # Because of performance implications, we don't actually return a wrapped method
  # here. Instead, we'll put an attribute on the method and replace the methods
  # later on when and if tracing is actually enabled.
  method.can_trace = True
  return method


ALL_PARSER_CLASSES = []

def parser_class(cls):
  ALL_PARSER_CLASSES.append(cls)
  return cls


class Parser(object):
  """A processing node, generated from a Grammar."""
  counter = 0

  def __init__(self, name):
    self.counter = Parser.counter
    Parser.counter += 1
    self.name = name

  def prefix_token_types(self):
    raise NotImplementedError("Return a list of token types that will be handled by this matcher.")

  def feed(self, token, context):
    raise NotImplementedError("Implement feed().")

  def children(self):
    return []

  def caption(self):
    return self.name or self.__class__.__name__

  def show(self):
    name = self.name if self.name else self.caption()
    return (name, self.children())


@parser_class
class FailParser(Parser):
  def __init__(self, name):
    Parser.__init__(self, name)

  def prefix_token_types(self):
    return []

  @trace_parser
  def parse(self, state):
    return state.make_failure(state.current_token(), 'This rule can never match')


@parser_class
class AtomParser(Parser):
  def __init__(self, token_type, capture, name):
    Parser.__init__(self, name)
    self.token_type = token_type
    self.capture = capture

  def prefix_token_types(self):
    return [self.token_type]

  @trace_parser
  def parse(self, state):
    token = state.current_token()
    if token.type != self.token_type:
      return state.make_failure(token, "Unexpected '%s', expected '%s'" % (token.value, self.token_type))
    return state.capture() if self.capture else state.skip()

  def caption(self):
    return Parser.caption(self) + '(%s)' % self.token_type


@parser_class
class NullParser(Parser):
  def prefix_token_types(self):
    return [AND_MORE]

  @trace_parser
  def parse(self, state):
    return state

  def caption(self):
    return self.name or 'NullParser [%d]' % self.counter


@parser_class
class PushValueParser(Parser):
  def __init__(self, value, name):
    Parser.__init__(self, name)
    self.value = value

  def prefix_token_types(self):
    return [AND_MORE]

  @trace_parser
  def parse(self, state):
    return state.push_value(self.value)

  def caption(self):
    return self.name or 'PushValueParser [%d]' % self.counter


@parser_class
class RecoverFailureParser(Parser):
  def __init__(self, inner, resync, name):
    Parser.__init__(self, name)
    self.inner = inner
    self.resync = resync

  def prefix_token_types(self):
    return self.inner.prefix_token_types()

  def caption(self):
    return self.name or 'RecoverFailure'

  @trace_parser
  def parse(self, state):
    ret = self.inner.parse(state)
    if not ret.is_success:
      x = self.resync.parse(state)
      # Only if resync succeeded, return that
      if x.is_success and x.tokens.i > ret.state.tokens.i:
        if Trace.enabled:
          Trace.report('Recovered.')
        return x

    return ret


@parser_class
class SequenceParser(Parser):
  def __init__(self, parsers, name):
    Parser.__init__(self, name)
    self.parsers = splat_parsers(SequenceParser, parsers)

    # Build a sequence of which positions allow backtracking, for quicker
    # lookup in the inner loop.
    self.parsers_and_bt = []
    allow_backtracking = True
    for parser in self.parsers:
      if parser is NO_BACKTRACKING_PARSER:
        allow_backtracking = False
      else:
        self.parsers_and_bt.append((parser, allow_backtracking))

  def prefix_token_types(self):
    ret = []
    for parser in self.parsers:
      ret.extend(parser.prefix_token_types())
      if AND_MORE not in ret:
        break
      ret.remove(AND_MORE)
    return ret

  @trace_parser
  def parse(self, state):
    for parser, allow_backtracking in self.parsers_and_bt:
      #assert state.is_success  # Asserts already have noticeable performance impact
      state = parser.parse(state)
      if not state.is_success:
        state.is_backtrackable_failure = allow_backtracking
        break
    return state

  def children(self):
    return self.parsers

  def caption(self):
    return self.name or 'SequenceParser [%d]' % self.counter


class NoBacktrackingMarker(object):
  def show(self):
    return 'No Backtracking', []

  def __repr__(self):
    return 'NO_BACKTRACKING_PARSER'


NO_BACKTRACKING_PARSER = NoBacktrackingMarker()


@parser_class
class AlternativesParser(Parser):
  def __init__(self, parsers, name):
    Parser.__init__(self, name)
    self.parsers = splat_parsers(AlternativesParser, parsers)
    self.jump = collections.defaultdict(list)
    self.build_jump_map()

  def build_jump_map(self):
    for parser in self.parsers:
      for token_type in parser.prefix_token_types():
        self.jump[token_type].append(parser)

    # Precompute for slight speed bonus
    self.describe_expected_tokens = ', '.join("'%s'" % t for t in self.prefix_token_types())

  def prefix_token_types(self):
    # Automatically covers AND_MORE
    return self.jump.keys()

  @trace_parser
  def parse(self, state):
    token = state.current_token()
    parsers = self.jump.get(token.type, []) + self.jump.get(AND_MORE, [])
    if not parsers:
      if AND_MORE in self.prefix_token_types():
        # It MIGHT be acceptable to parse nothing here. Try returning the input state and trying there.
        return state

      return state.make_failure(token, "Unexpected '%s', expected one of %s" % (token.value, self.describe_expected_tokens))

    errors = []
    for parser in parsers:
      # Enable backtracking over our alternatives
      ret = parser.parse(state)
      if not ret.is_backtrackable_failure:  # Either success or not backtrackable
        return ret
      errors.append(ret)

    # Return the error that parses the farthest (idea happily borrowed from pyparsing :)
    errors.sort(key=lambda x: x.token.span)
    return errors[-1]

  def children(self):
    return self.parsers

  def caption(self):
    return self.name or 'Alternatives [%d]' % self.counter


@parser_class
class OptionalParser(Parser):
  def __init__(self, inner, name, missing_default=NO_DEFAULT):
    Parser.__init__(self, name)
    self.inner = inner
    self.inner_parse = inner.parse
    self.missing_default = missing_default

  def prefix_token_types(self):
    return self.inner.prefix_token_types() + [AND_MORE]

  @trace_parser
  def parse(self, state):
    ret = self.inner_parse(state)
    if not ret.is_success:
      if not ret.is_backtrackable_failure:
        return ret

      # Attach the failure to the successful parse
      if self.missing_default is not NO_DEFAULT:
        state = state.push_value(self.missing_default)
      return state.with_failure(ret)
    return ret

  def children(self):
    return [self.inner]

  def caption(self):
    return self.name or 'Optional [%d]' % (self.counter)


@parser_class
class CountParser(Parser):
  def __init__(self, inner, min_count, max_count, name, missing_default=NO_DEFAULT):
    Parser.__init__(self, name)
    self.inner = inner
    self.min_count = min_count
    self.max_count = max_count
    self.missing_default = missing_default

  def prefix_token_types(self):
    return self.inner.prefix_token_types() + ([AND_MORE] if self.min_count == 0 else [])

  @trace_parser
  def parse(self, state):
    #assert state.is_success
    i = 0

    # These are required
    while i < self.min_count:
      state = self.inner.parse(state)
      if not state.is_success:
        return state
      i += 1

    # The rest is optional
    while i < self.max_count or self.max_count is None:
      #assert state.is_success

      # Enable backtracking over our alternatives
      ret = self.inner.parse(state)

      # If not a successful parse, we'll assume the missing value and continue
      if not ret.is_success:
        if not ret.is_backtrackable_failure:
          return ret

        if i == 0 and self.missing_default is not NO_DEFAULT:
          state = state.push_value(self.missing_default)

        return state

      state = ret
      i += 1

    return state

  def children(self):
    return [self.inner]

  def caption(self):
    return self.name or 'Count(%s..%s) [%d]' % (self.min_count, self.max_count, self.counter)


@parser_class
class EndOfFileParser(Parser):
  def prefix_token_types(self):
    return [END_OF_FILE]

  @trace_parser
  def parse(self, state):
    token = state.current_token()
    if token.type != END_OF_FILE:
      return state.make_failure(token, "Unexpected '%s', expected end of file" % token.value)
    # Don't advance (otherwise the state becomes unprintable)
    return state

  def caption(self):
    return self.name or 'EOF [%d]' % self.counter


@parser_class
class CapturingParser(Parser):
  def __init__(self, inner, action, name):
    Parser.__init__(self, name)
    self.inner = inner
    self.action = action

  def prefix_token_types(self):
    return self.inner.prefix_token_types()

  @trace_parser
  def parse(self, state):
    # Creating a copy of the state object happens a lot. It's cheaper
    # to pass the old one and remember the index we got to.
    old_value_count = len(state.values)
    start_token_i = state.tokens.i

    ret = self.inner.parse(state)
    if not ret.is_success:
      return ret

    # We get the span in an ugly way (by indexes) to squeeze out some extra milliseconds
    end_token_i = max(ret.tokens.i - 1, start_token_i)
    state_tokens_lazy_list = state.tokens.lazy_list
    whole_span = state_tokens_lazy_list[start_token_i].span.until(state_tokens_lazy_list[end_token_i].span)

    # Old values with dispatch result appended, new location
    try:
      new_value = self.action(whole_span, *ret.values[old_value_count:])
    except TypeError as e:
      raise TypeError('While calling %r: %s' % (self.action, e))

    if Trace.enabled:
      Trace.report('Captured: %r', new_value)

    ret.values[old_value_count:] = values_from_value(new_value)
    return ret

  def children(self):
    return [self.inner]


@parser_class
class AnythingExceptParser(Parser):
  def __init__(self, token_types, capture, name):
    Parser.__init__(self, name)
    self.capture = capture
    self.token_types = token_types

  def prefix_token_types(self):
    return [AND_MORE]

  @trace_parser
  def parse(self, state):
    token = state.current_token()
    if token.type in self.token_types:
      return state.make_failure(token, "Got token '%s', not allowed here" % token.value)
    return state.capture() if self.capture else state.skip()


@parser_class
class ForwardParser(Parser):
  def __init__(self, name):
    Parser.__init__(self, name)
    self.inner = None

  def prefix_token_types(self):
    #assert self.inner
    return self.inner.prefix_token_types()

  def set(self, inner):
    assert self.inner is None
    self.inner = inner

  @trace_parser
  def parse(self, state):
    return self.inner.parse(state)

  def children(self):
    return [self.inner]

  def caption(self):
    return self.inner.caption()


class Splat(collections.namedtuple('Splat', ['values'])):
  """A marker type that exists to expand into multiple arguments to a capturing function."""
  pass


#--------------------------------------------------------------------------------
#  PARSING INFRA


class ParseState(object):
  # Turn is_success into an attribute instead of a method call,
  # for slight speed improvement.
  #
  # def is_success(self):
    # raise NotImplementedError("Oei")

  def show(self):
    raise NotImplementedError("Oei")


CALLS = collections.Counter()

class ParseSuccess(ParseState):
  is_success = True
  is_backtrackable_failure = False

  def __init__(self, tokens, values, furthest_failure):
    #assert isinstance(values, list)
    self.tokens = tokens
    self.values = values
    self.furthest_failure = furthest_failure

    if Trace.enabled:
      Trace.success(self)

    # Copy forwarded functions for speed
    self.current_token = self.tokens.current
    self.previous_token = self.tokens.previous

  def with_values(self, values):
    return ParseSuccess(self.tokens, values, self.furthest_failure)

  # def current_token(self):
    # return self.tokens.current()

  # def previous_token(self):
    # return self.tokens.previous()

  def capture(self):
    token = self.tokens.current()
    return ParseSuccess(self.tokens.advanced(), self.values + [token], self.furthest_failure)

  def push_value(self, value):
    return ParseSuccess(self.tokens, self.values + values_from_value(value), self.furthest_failure)

  def skip(self):
    return ParseSuccess(self.tokens.advanced(), self.values, self.furthest_failure)

  def with_failure(self, furthest_failure):
    #assert isinstance(furthest_failure, ParseFailure)
    return ParseSuccess(self.tokens, self.values, furthest_failure)

  def absorb_failure(self, state):
    if isinstance(state, ParseSuccess):
      return self
    #assert isinstance(state, ParseFailure)
    if self.furthest_failure is not None and self.furthest_failure.token.span > state.token.span:
      return self
    return self.with_failure(state)

  def show(self):
    return 'At token %s, %r' % (self.tokens.i, self.tokens.current())

  def show_success(self):
    return 'Success at token %s, %r' % (self.tokens.i - 1, self.tokens.previous())

  def show_position(self, message='here'):
    return '\n'.join(self.current_token().span.annotated_source(message))

  def make_failure(self, token, error):
    """Return the farthest error between the one on this success state and a new one."""
    fresh = ParseFailure(token, error, self)
    if not self.furthest_failure:
      return fresh
    return self.furthest_failure.furthest(fresh)


def values_from_value(value):
  if isinstance(value, Splat):
    return value.values
  else:
    return [value]


class ParseFailure(ParseState):
  is_success = False

  def __init__(self, token, error, state):
    self.token = token
    self.error = error
    self.state = state
    self.is_backtrackable_failure = True

    if Trace.enabled:
      Trace.failure(self)

  def furthest(self, other):
    if self.token.span > other.token.span:
      return self
    else:
      return other

  def recover(self):
    return self.state

  def show(self):
    return '\n'.join(self.token.span.annotated_source(self.error))

  def __repr__(self):
    return self.show()


class TokenStream(object):
  def __init__(self, lazy_list, i, file):
    self.lazy_list = lazy_list
    self.i = i
    self.file = file

  def current(self):
    return self.lazy_list[self.i]

  def previous(self):
    #assert self.i > 0
    return self.lazy_list[self.i - 1]

  def advanced(self):
    return TokenStream(self.lazy_list, self.i + 1, self.file)

  def show(self):
    return 'token %s, %r' % (self.i, self.current())


class EagerList(list):
  def __init__(self, iterable):
    super(EagerList, self).__init__(iterable)
    self.len_list = len(iterable)

  def past_end(self, i):
    return i >= self.len_list


class LazyList(object):
  """A list based on a generator that allows access to already-generated elements.

  Alternative name: memoizing generator.
  """
  def __init__(self, iterable):
    self.iterator = iter(iterable)
    self.next_value = None
    self._step()
    self.reified = []
    self.len_reified = 0
    self.eof = False

  def past_end(self, i):
    #assert i <= len(self.reified)  # Can't answer question otherwise
    return i == self.len_reified and self.eof

  def _step(self):
    try:
      self.next_value = self.iterator.next()
    except StopIteration:
      self.eof = True

  def __getitem__(self, i):
    if i > self.len_reified:
      raise RuntimeError('Must sequentially advance in lazy list')
    if i == self.len_reified:
      if self.eof:
        raise IndexError(i)
      self.reified.append(self.next_value)
      self.len_reified += 1
      self._step()
    return self.reified[i]


class Trace(object):
  current_trace_stack = []
  enabled = False
  names_only = False

  def __init__(self, parser, state):
    self.parser = parser
    self.state = state
    self.pushed = False

  def __enter__(self):
    if Trace.enabled and (not Trace.names_only or self.parser.name):
      print('  ' * len(Trace.current_trace_stack) + '%s  |  %s' % (self.parser.show()[0], self.state.show()))
      self.pushed = True
      Trace.current_trace_stack.append(self)
    return self

  def __exit__(self, value, type, tb):
    if self.pushed:
      Trace.current_trace_stack.pop()

  @staticmethod
  def enable(enabled, names_only=False):
    """Enable parse tree tracing.

    We replace the parsing methods that can be traced when tracing is enabled,
    so that normal operation comes with minimal performance penalty.
    """
    Trace.enabled = enabled
    Trace.names_only = names_only

    import inspect
    for klass in ALL_PARSER_CLASSES:
      for name, method in inspect.getmembers(klass, predicate=inspect.ismethod):
        if enabled and getattr(method, 'can_trace', False):
          setattr(klass, name, make_tracing_method(method))
        elif not enabled and getattr(method, 'original_method', None):
          setattr(klass, name, method.original_method)

  @staticmethod
  def report(message, *args):
    prefix = '  ' * len(Trace.current_trace_stack) + '> '
    s = message % args
    print(prefix + s.replace('\n', '\n' + prefix))

  @staticmethod
  def success(state):
    print('  ' * len(Trace.current_trace_stack) + '*** ' + state.show_success())

  @staticmethod
  def failure(state):
    prefix = '  ' * len(Trace.current_trace_stack) + '!!! '
    print(prefix + state.show().replace('\n', '\n' + prefix))


def make_tracing_method(method):
  @functools.wraps(method)
  def decorated(self, state):
    with Trace(self, state):
      return method(self, state)
  decorated.original_method = method
  return decorated


def make_parser(grammar):
  return SequenceParser([
    grammar.make_parser(),
    EndOfFileParser(None)
    ], None)


def parse_all(parser, tokens, file):
  if isinstance(tokens, list):
    the_list = EagerList(tokens)
  else:
    the_list = LazyList(tokens)

  tokens = TokenStream(the_list, 0, file)
  state = parser.parse(ParseSuccess(tokens, [], furthest_failure=None))
  if not state.is_success:
    raise ParseError(state.token.span, state.error)

  return state.values


def print_parser(parser, stream):
  uniques = set()
  stack = [(0, parser)]
  while stack:
    depth, top = stack.pop()
    caption, children = top.show()
    prefix = '|   ' * max(0, depth - 1) + ('|---' if depth > 0 else '')
    suffix = ' (**recursion**)' if top in uniques else ''

    print('%s%s%s' % (prefix, caption, suffix))

    if top not in uniques:
      uniques.add(top)
      for child in reversed(children):
        stack.append((depth + 1, child))


def flatmap(f, items):
  return itertools.chain.from_iterable(itertools.imap(f, items))


def splat_parsers(klass, parsers):
  # Splat out other SequenceParsers
  return list(flatmap(lambda p: p.parsers if isinstance(p, klass) else [p], parsers))


def find_line_context(text, start):
  """From a text and an offset, find the line nr, the line contents and the index in the line.

  Line nr is 1-based
  Colum numbers is 0-based.
  """
  line_nr = 1
  line_start = 0
  for ix in all_newlines(text):
    if ix >= start:
      break
    line_start = ix + 1
    line_nr += 1

  line_end = text.find('\n', start)
  if line_end == -1:
    line_end = len(text)

  return line_nr, text[line_start:line_end], start - line_start


def all_newlines(s):
  """Return the indexes of all newlines in string s."""
  for m in re.compile('\n').finditer(s):
    yield m.start()


def linecol_to_index(s, line, col):
  """Return string index from 1-based line, 0-based col.

  Returns:
    string index of indicated position, or length of string if index past end.
  """
  line -= 1
  line_start = 0
  for ix in all_newlines(s):
    if line == 0:
      break
    line -= 1
    line_start = ix + 1

  # Ate all newlines
  if line != 0:
    return len(s)

  return line_start + col
