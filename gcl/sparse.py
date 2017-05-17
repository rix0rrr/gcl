# coding=utf-8
"""sparse -- A faster parser combinator library for Python.

Written when we outgrew the performance concerns of pyparsing:

  - Uses a tokenizer: parses tokens instead of characters.
  - Does not use exceptions to report parse failures.
  - Uses jump tables to speed up parsing.

The tokenizer is based on regex module tricks from:

    http://lucumr.pocoo.org/2015/11/18/pythons-hidden-re-gems/
"""
from __future__ import unicode_literals

import collections
import itertools
import copy
import re
import sys

from sre_parse import Pattern, SubPattern, parse
from sre_compile import compile as sre_compile
from sre_constants import BRANCH, SUBPATTERN


class Symbol(object):
  def __init__(self, name):
    self.name = name

  def __repr__(self):
    return self.name


END_OF_FILE = Symbol('end-of-file')
AND_MORE = Symbol('and-more')


class ParseError(RuntimeError):
  def __init__(self, loc, message):
    RuntimeError.__init__(self, message)
    self.loc = loc

  def add_context(self, filename, contents):
    line_nr, line_text, line_offset = find_line_context(contents, self.loc[0])
    error_message = '%s:%d: %s' % (filename, line_nr + 1, self.message)
    uw = max(3, self.loc[1] - self.loc[0])
    underline = ' '  * line_offset + '^' * uw + ' here'

    return ParseError(self.loc,
                      error_message + '\n' + line_text + '\n' + underline)


#--------------------------------------------------------------------------------
#  TOKENIZER


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

  def scan(self, string, skip=False):
    sc = self._scanner(string)

    match = None
    for match in iter(sc.search if skip else sc.match, None):
      rule = self.rules[match.lastindex - 1]
      yield rule.token_type, match, rule.post_processor

    if match.end() < len(string):
      span = (match.end(), match.end() + 1)
      raise ParseError(span, 'Unable to match token at %s' % (string[match.end():match.end() + 10] + '...'))

  def tokenize(self, string):
    for token, match, proc in self.scan(string):
      if token == 'whitespace' or token == 'comment':
        continue
      if token == 'keyword' or token == 'symbol':
        token = match.group()
      value = proc(match.group()) if proc else match.group()
      yield Token(token, value, match.span())


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
    return StopBacktrackingParser(self.name)

NO_BACKTRACKING = StopBacktrackingGrammar()


class CapturingGrammar(Grammar):
  def __init__(self, inner, action):
    Grammar.__init__(self)
    self.inner = inner.take_reference()
    self.action = action

  def make_parser(self):
    return CapturingParser(self.inner.make_parser(), self.action, self.name)


class T(Grammar):
  def __init__(self, token_type, capture=True):
    Grammar.__init__(self)
    self.token_type = token_type
    self.capture = capture

  def suppress(self):
    return T(self.token_type, capture=False)

  def make_parser(self):
    return AtomParser(self.token_type, self.capture, self.name)


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
  def __init__(self, inner, default=None):
    Grammar.__init__(self)
    self.inner = inner.take_reference()
    self.default = default

  def make_parser(self):
    return OptionalParser(self.inner.make_parser(), self.name, missing_default=self.default)
    return CountParser(self.inner.make_parser(), 0, 1, self.name, missing_default=self.default)


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
      return CapturingParser(BacktrackingScopeParser(self.grammar.make_parser(), None), self.action, self.name)
    else:
      return BacktrackingScopeParser(self.grammar.make_parser(), self.name)


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


@parser_class
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
    raise NotImplementedError("Umm.")

  def show(self):
    name = self.name if self.name else self.caption()
    return (name, self.children())


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
      return state.make_failure(token, "Unexpected '%s', expected '%s'" % (token.type, self.token_type), can_backtrack=state.allow_backtracking)
    return state.capture() if self.capture else state.skip()

  def caption(self):
    return 'AtomParser(%s) [%d]' % (self.token_type, self.counter)


@parser_class
class NullParser(Parser):
  def prefix_token_types(self):
    return [AND_MORE]

  @trace_parser
  def parse(self, state):
    return state

  def caption(self):
    return 'NullParser [%d]' % self.counter


@parser_class
class StopBacktrackingParser(Parser):
  def prefix_token_types(self):
    return [AND_MORE]

  @trace_parser
  def parse(self, state):
    state.allow_backtracking = False
    return state

  def caption(self):
    return 'StopBacktrackingParser [%d]' % self.counter


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
    return 'PushValueParser [%d]' % self.counter


@parser_class
class SequenceParser(Parser):
  def __init__(self, parsers, name):
    Parser.__init__(self, name)
    self.parsers = splat_parsers(SequenceParser, parsers)

  def prefix_token_types(self):
    ret = self.parsers[0].prefix_token_types()
    while AND_MORE in ret:
      for parser in self.parsers[1:]:
        if parser is not NO_BACKTRACKING:
          ret.remove(AND_MORE)
          ret.extend(parser.prefix_token_types())
    return ret

  @trace_parser
  def parse(self, state):
    for parser in self.parsers:
      assert state.is_success()
      state = parser.parse(state)
      if not state.is_success():
        break
    return state

  def children(self):
    return self.parsers

  def caption(self):
    return 'SequenceParser [%d]' % self.counter


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

  def prefix_token_types(self):
    # Automatically covers AND_MORE
    return self.jump.keys()

  @trace_parser
  def parse(self, state):
    token = state.current_token()
    parsers = self.jump.get(token.type, [])
    if not parsers:
      if AND_MORE in self.prefix_token_types():
        # It MIGHT be acceptable to parse nothing here. Try returning the input state and trying there.
        return state

      return state.make_failure(token, "Unexpected '%s', expected one of %s" % (token.type, ', '.join("'%s'" % t for t in self.prefix_token_types())), can_backtrack=state.allow_backtracking)

    errors = []
    for parser in parsers:
      # Enable backtracking over our alternatives
      state.allow_backtracking = True
      ret = parser.parse(state)
      if not ret.is_backtrackable_failure():  # Either success or not backtrackable
        return ret
      errors.append(ret)

    # Return the error that parses the farthest (idea happily borrowed from pyparsing :)
    errors.sort(key=lambda x: x.token.span)
    return errors[-1]

  def children(self):
    return self.parsers

  def caption(self):
    return 'Alternatives [%d]' % self.counter


@parser_class
class OptionalParser(Parser):
  def __init__(self, inner, name, missing_default=None):
    Parser.__init__(self, name)
    self.inner = inner
    self.missing_default = missing_default

  def prefix_token_types(self):
    return self.inner.prefix_token_types() + [AND_MORE]

  @trace_parser
  def parse(self, state):
    state.allow_backtracking = True
    ret = self.inner.parse(state)
    if ret.is_backtrackable_failure():
      # Attach the failure to the successful parse
      return (state.push_value(self.missing_default) if self.missing_default is not None else state).with_failure(ret)
    return ret

  def children(self):
    return [self.inner]

  def caption(self):
    return 'Optional [%d]' % (self.counter)


@parser_class
class CountParser(Parser):
  def __init__(self, inner, min_count, max_count, name, missing_default=None):
    Parser.__init__(self, name)
    self.inner = inner
    self.min_count = min_count
    self.max_count = max_count
    self.missing_default = missing_default

  def prefix_token_types(self):
    return self.inner.prefix_token_types() + ([AND_MORE] if self.min_count == 0 else [])

  @trace_parser
  def parse(self, state):
    assert state.is_success()
    i = 0

    # These are required
    while i < self.min_count:
      state = self.inner.parse(state)
      if not state.is_success():
        return state
      i += 1

    # The rest is optional
    while i < self.max_count or self.max_count is None:
      assert state.is_success()

      # Enable backtracking over our alternatives
      state.allow_backtracking = True
      ret = self.inner.parse(state)

      # If not a successful parse, we'll assume the missing value and continue
      if ret.is_backtrackable_failure():
        if i == 0 and self.missing_default is not None:
          return state.push_value(self.missing_default)
        return state
      if not ret.is_success():
        return ret

      state = ret
      i += 1

    return state

  def caption(self):
    return [self.inner]

  def caption(self):
    return 'Count(%s..%s) [%d]' % (self.min_count, self.max_count, self.counter)


@parser_class
class EndOfFileParser(Parser):
  def prefix_token_types(self):
    return [END_OF_FILE]

  @trace_parser
  def parse(self, state):
    token = state.current_token()
    if token.type != END_OF_FILE:
      return state.make_failure(token, "Unexpected '%s', expected end of file" % token.type, can_backtrack=state.allow_backtracking)
    # Don't advance (otherwise the state becomes unprintable)
    return state

  def caption(self):
    return 'EOF [%d]' % self.counter


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

    ret = self.inner.parse(state)
    if not ret.is_success():
      return ret

    # Old values with dispatch result appended, new location
    new_value = self.action(*ret.values[old_value_count:])
    if Trace.enabled:
      Trace.report('Captured: %r', new_value)

    ret.values[old_value_count:] = [new_value]
    return ret

  def children(self):
    return [self.inner]

  def caption(self):
    return 'Capture [%d]' % self.counter


@parser_class
class ForwardParser(Parser):
  def __init__(self, name):
    Parser.__init__(self, name)
    self.inner = None

  def prefix_token_types(self):
    assert self.inner
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
    return 'Forward [%d]' % self.counter


@parser_class
class BacktrackingScopeParser(Parser):
  def __init__(self, inner, name):
    Parser.__init__(self, name)
    self.inner = inner

  def prefix_token_types(self):
    return self.inner.prefix_token_types()

  @trace_parser
  def parse(self, state):
    # Return whatever the inner parser produces, but reset the backtracking state
    # to whatever it was coming into this parser. Doesn't need a new object creation.
    ret = self.inner.parse(state)
    ret.can_backtrack = state.allow_backtracking
    return ret

  def children(self):
    return [self.inner]

  def caption(self):
    return 'BacktrackingScopeParser'


#--------------------------------------------------------------------------------
#  PARSING INFRA


class ParseState(object):
  def is_success(self):
    raise NotImplementedError("Oei")

  def show(self):
    raise NotImplementedError("Oei")


CALLS = collections.Counter()

class ParseSuccess(ParseState):
  __slots__ = ('tokens', 'values', 'allow_backtracking', 'furthest_failure')

  def __init__(self, tokens, values, allow_backtracking=True, furthest_failure=None):
    assert isinstance(values, list)
    self.tokens = tokens
    self.values = values
    self.allow_backtracking = allow_backtracking
    self.furthest_failure = furthest_failure

    import traceback
    stack = traceback.extract_stack()
    callers_linenos = tuple(['%s:%s' % (s[1], s[3]) for s in stack[-5:-1]])
    CALLS.update([callers_linenos])

    if Trace.enabled:
      Trace.success(self)

  def with_values(self, values):
    return ParseSuccess(self.tokens, values, self.allow_backtracking, self.furthest_failure)

  def current_token(self):
    return self.tokens.current()

  def capture(self):
    token = self.tokens.current()
    return ParseSuccess(self.tokens.advanced(), self.values + [token.value], self.allow_backtracking, self.furthest_failure)

  def push_value(self, value):
    return ParseSuccess(self.tokens, self.values + [value], self.allow_backtracking, self.furthest_failure)

  def skip(self):
    return ParseSuccess(self.tokens.advanced(), self.values, self.allow_backtracking, self.furthest_failure)

  def with_backtracking(self, allow_backtracking):
    return ParseSuccess(self.tokens, self.values, allow_backtracking, self.furthest_failure)

  def with_failure(self, furthest_failure):
    assert isinstance(furthest_failure, ParseFailure)
    return ParseSuccess(self.tokens, self.values, self.allow_backtracking, furthest_failure)

  def is_success(self):
    return True

  def is_backtrackable_failure(self):
    return False

  def absorb_failure(self, state):
    if isinstance(state, ParseSuccess):
      return self
    assert isinstance(state, ParseFailure)
    if self.furthest_failure is not None and self.furthest_failure.token.span > state.token.span:
      return self
    return self.with_failure(state)

  def show(self):
    return '%s%s' % (self.tokens.show(), ' (b)' if self.allow_backtracking else '')

  def make_failure(self, *args, **kwargs):
    """Return the farthest error between the one on this success state and a new one."""
    fresh = ParseFailure(*args, **kwargs)
    if not self.furthest_failure:
      return fresh
    return self.furthest_failure.furthest(fresh)


class ParseFailure(ParseState):
  def __init__(self, token, error, can_backtrack=True):
    self.token = token
    self.error = error
    self.can_backtrack = can_backtrack

    if Trace.enabled:
      Trace.failure(self)

  def is_success(self):
    return False

  def is_backtrackable_failure(self):
    return self.can_backtrack

  def with_backtracking(self, can_backtrack):
    return ParseFailure(self.token, self.error, can_backtrack=can_backtrack)

  def furthest(self, other):
    if self.token.span > other.token.span:
      return self
    else:
      return other

  def show(self):
    return '%s: %s%s' % (self.token, self.error, ' (b)' if self.can_backtrack else '')

  def __repr__(self):
    return self.show()


class TokenStream(object):
  __slots__ = ('lazy_list', 'i')

  def __init__(self, lazy_list, i):
    self.lazy_list = lazy_list
    self.i = i

  def current(self):
    tok = self.lazy_list[self.i] if not self.lazy_list.past_end(self.i) else self.make_eof_token(self.i)
    assert isinstance(tok, Token)
    return tok

  def make_eof_token(self, i):
    span = (0, 0)
    if i > 0:
      prev = self.lazy_list[i-1]
      span = (prev.span[1], prev.span[1] + 1)
    return Token(END_OF_FILE, '', span)

  def advanced(self):
    return TokenStream(self.lazy_list, self.i + 1)

  def show(self):
    return str(self.current())


class LazyList(object):
  """A list based on a generator that allows access to already-generated elements.

  Alternative name: memoizing generator.
  """
  def __init__(self, iterable):
    self.iterator = iter(iterable)
    self.next_value = None
    self._step()
    self.reified = []
    self.eof = False

  def past_end(self, i):
    assert i <= len(self.reified)  # Can't answer question otherwise
    return i == len(self.reified) and self.eof

  def _step(self):
    try:
      self.next_value = self.iterator.next()
    except StopIteration:
      self.eof = True

  def __getitem__(self, i):
    if i > len(self.reified):
      raise RuntimeError('Must sequentially advance in lazy list')
    if i == len(self.reified):
      if self.eof:
        raise IndexError(i)
      self.reified.append(self.next_value)
      self._step()
    return self.reified[i]


class Trace(object):
  trace_stack = []
  enabled = False
  names_only = False

  def __init__(self, parser, state):
    self.parser = parser
    self.state = state

  def __enter__(self):
    if Trace.enabled and (not Trace.names_only or self.parser.name):
      print 'yay'
      print('  ' * len(Trace.trace_stack) + '%s  |  %s' % (self.parser.show()[0], self.state.show()))
      Trace.trace_stack.append(self)
    return self

  def __exit__(self, value, type, tb):
    if Trace.enabled and (not Trace.names_only or self.parser.name):
      Trace.trace_stack.pop()

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
          klass.name = make_tracing_method(method)
        elif not enabled and getattr(method, 'original_method', None):
          klass.name = method.original_method

  @staticmethod
  def report(message, *args):
    if Trace.enabled:
      print('  ' * len(Trace.trace_stack) + (message % args))

  @staticmethod
  def success(state):
    pass
    # if Trace.enabled:
      # print '  ' * Trace.indent + state.show()

  @staticmethod
  def failure(state):
    if Trace.enabled:
      print('  ' * len(Trace.trace_stack) + '!!! %s' % state.show())


def make_tracing_method(method):
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


def parse_all(parser, tokens):
  tokens = TokenStream(LazyList(tokens), 0)

  state = parser.parse(ParseSuccess(tokens, []))
  if not state.is_success():
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
  """From a text and an offset, find the line nr, the line contents and the index in the line."""
  m = None
  line_nr = 0
  for line_nr, m in enumerate(re.compile('\n').finditer(text, 0, start), 1):
    pass

  line_start = m.start() + 1 if m else 0
  line_end = text.find('\n', start)
  if line_end == -1:
    line_end = len(text)

  return line_nr, text[line_start:line_end], start - line_start
