# coding=utf-8
"""sparse -- A faster parser combinator library for Python.

Written when we outgrew the performance concerns of pyparsing:

  - Uses a tokenizer: parses tokens instead of characters.
  - Does not use exceptions to report parse failures.
  - Uses jump tables to speed up parsing (forms a pushdown automaton
    from the grammar).

The tokenizer is based on regex module tricks from:

    http://lucumr.pocoo.org/2015/11/18/pythons-hidden-re-gems/
"""
from __future__ import unicode_literals

import collections
import copy
import re

from sre_parse import Pattern, SubPattern, parse
from sre_compile import compile as sre_compile
from sre_constants import BRANCH, SUBPATTERN


class Epsilon(object):
  def __str__(self):
    return '_'

EPSILON = Epsilon()


class ParseError(RuntimeError):
  def __init__(self, loc, message):
    RuntimeError.__init__(self, message)
    self.loc = loc


class Grammar(object):
  def __init__(self):
    self.refcount = 0
    self.transitions_into_routine = None

  def transitions(self, into):
    if self.refcount <= 1:   # 0 or 1, 0 for top-level grammar
      # Simple transition (validate that we only get called once or w/ the same target)
      return self.make_transitions(into)

    if self.transitions_into_routine is None:
      # Make a call/ret subautomaton
      #
      #   +-------------------+
      #   |      A     B      |
      #   |   .---->O---->O   |
      #   |   |         (ret) |
      #   +-------------------+
      #     \ |             /
      #       |\         /
      #       |   \   /
      #-------+- - >O----------->O
      #    eps    into    C
      #   (call)
      self.transitions_into_routine = self.make_transitions(ReturnState())

    # Make transition
    entry_state = State(self.transitions_into_routine)
    return [Transition(EPSILON, entry_state, push_retstate=into)]

  def action(self, action):
    return CapturingGrammar(self, action)

  def take_reference(self):
    self.refcount += 1
    return self

  def __add__(self, rhs):
    return And(self, rhs)

  def __or__(self, rhs):
    return Either(self, rhs)


class CapturingGrammar(Grammar):
  def __init__(self, inner, action):
    Grammar.__init__(self)
    self.inner = inner.take_reference()
    self.action = action

  def make_transitions(self, into):
    out_state = State([Transition(EPSILON, into, dispatch_captures=self.action)])
    trans = self.inner.transitions(out_state)
    for t in trans:
      t.push_captures = True
    return trans


class Empty(Grammar):
  def make_transitions(self, into):
    return [Transition(EPSILON, into)]


class T(Grammar):
  def __init__(self, token_type, capture=True):
    Grammar.__init__(self)
    self.token_type = token_type
    self.capture = capture

  def suppress(self):
    return T(self.token_type, capture=False)

  def make_transitions(self, into):
    return [Transition(self.token_type, into, capture_token=self.capture)]


def Q(token_type):
  """Quiet token."""
  return T(token_type, capture=False)


class And(Grammar):
  def __init__(self, left, right):
    assert(isinstance(left, Grammar) and isinstance(right, Grammar))
    Grammar.__init__(self)
    self.left = left.take_reference()
    self.right = right.take_reference()

  def make_transitions(self, into):
    return self.left.transitions(State(self.right.transitions(into)))


class Optional(Grammar):
  def __init__(self, inner, default=None):
    Grammar.__init__(self)
    self.inner = inner.take_reference()
    self.default = default

  def make_transitions(self, into):
    skip_transitions = [t.capture_value(self.default) for t in into.transitions()]
    return self.inner.transitions(into) + skip_transitions


class Either(Grammar):
  def __init__(self, left, right):
    assert(isinstance(left, Grammar) and isinstance(right, Grammar))
    Grammar.__init__(self)
    self.left = left.take_reference()
    self.right = right.take_reference()

  def make_transitions(self, into):
    return self.left.transitions(into) + self.right.transitions(into)


class ZeroOrMore(Grammar):
  def __init__(self, inner):
    assert isinstance(inner, Grammar)
    Grammar.__init__(self)
    self.inner = inner.take_reference()

  def make_transitions(self, into):
    inner_state = State(into.transitions())
    inner_state.add_transitions(self.inner.transitions(inner_state))
    return into.transitions() + [Transition(EPSILON, inner_state)]


def delimited_list(opener, expr, sep, closer):
  """Delimited list with optional terminating separator."""
  return opener + Optional(expr + ZeroOrMore(sep + expr) + Optional(sep)) + closer


def parenthesized(expr):
  """Parenthesized version of the inner expression."""
  return Q('(') + expr + Q(')')


class Forward(Grammar):
  counter = 0

  def __init__(self):
    Grammar.__init__(self)
    self.inner = None
    self.recursing = False
    self.instance = Forward.counter
    Forward.counter += 1

  def set(self, inner):
    assert isinstance(inner, Grammar)
    self.inner = inner.take_reference()

  def make_transitions(self, into):
    if not self.inner:
      raise RuntimeError('Forward grammar never initialized')

    if self.recursing:
      # Add a transition from out_state to into(2)
      self.out_state.add_transition(Transition(EPSILON, into, POP, self.instance))
      return [Transition(EPSILON, self.in_state, PUSH, self.instance)]

    # For the following grammar:
    #
    #   EXPR := a | '(' EXPR ')'
    #
    # We're going to make the following automaton:
    #
    #  eps  in             a                out eps
    # ----->O------------------------------->O----->O into(1)
    #       | /\                          /\ |
    #     ( |  |                          |  |  eps/POP
    #       V  |  eps/PUSH             (  |  V
    #       O /                            \ O  into(2)
    #
    # I.e., inner and outer entering states of the Forward
    # are going to point to each other with an epsilon transition.


    self.out_state = State([Transition(EPSILON, into)])
    self.in_state = State()

    # Next call might recurse so in_state must already be set
    self.recursing = True
    self.in_state.add_transitions(self.inner.transitions(self.out_state))
    self.recursing = False

    return [Transition(EPSILON, self.in_state)]


def P(name, grammar, action=None):
  # FIXME: Add name to grammar, make sure name is reported in error messages.
  if action:
    return grammar.action(action)
  else:
    return grammar


def delimitedList(expr, sep):
  """Delimited list with trailing separator."""
  return Optional(expr + ZeroOrMore(sep + expr) + Optional(sep))


class State(object):
  def __init__(self, transitions=[], tag=None):
    self.transition_map = collections.defaultdict(set)
    self.returning = False
    self.terminating = False
    self.add_transitions(transitions)

  def transitions(self):
    return [t for ts in self.transition_map.values() for t in ts]

  def make_transitions(self):
    ret = []
    for ts in self.transition_map.values():
      ret.extend(ts)
    return ret

  def add_transitions(self, transitions):
    for trans in transitions:
      self.add_transition(trans)

  def remove_transition(self, transition):
    lst = self.transition_map[transition.token_type]
    if transition in lst:
      lst.remove(transition)

  def add_transition(self, trans):
    assert isinstance(trans, Transition)
    self.transition_map[trans.token_type].add(trans)

  def report_error(self, token):
    """Report an error, but only if we have useful transitions."""
    if self.transition_map:
      raise ParseError(token.span, 'Unexpected %s, expected one of %s' % (token.type, ', '.join(repr(k)
                                                                                                for k, v in self.transition_map.items()
                                                                                                if k != EPSILON and v)))

class EndState(State):
  """Terminating state of the entire grammar."""
  def __init__(self):
    self.transition_map = collections.defaultdict(list)
    self.returning = False
    self.terminating = True


class ReturnState(State):
  """Terminating state of sub-automaton."""
  def __init__(self):
    self.transition_map = collections.defaultdict(list)
    self.returning = True
    self.terminating = False


NONE, PUSH, POP = range(3)


def hash_all(obj):
  h = 0
  for x in obj.__dict__.values():
    h = 101 * h + hash(x)
  return h


def eq_all(obj, other):
  return obj.__dict__ == other.__dict__


class Transition(object):
  def __init__(self, token_type, target_state, stack_action=NONE, stack_value=None, push_captures=False,
      push_retstate=None, dispatch_captures=None, capture_token=True, fixed_capture=None):

    # Token type that this transition consumes, or EPSILON, and where it leads
    self.token_type = token_type
    self.target_state = target_state

    # What happens to the pushdown stack when this transition is taken
    # (either PUSH or POP a specific value)
    self.stack_action = stack_action
    self.stack_value = stack_value

    # Push a state onto the "call stack" of states
    self.push_retstate = push_retstate

    # Capture-related fields:
    # - start a new capture frame?
    # - capture the consumed token onto the capture frame, or capture a fixed token onto the frame?
    # - dispatch current capture frame to a handler?
    self.push_captures = push_captures
    self.capture_token = capture_token
    self.fixed_capture = fixed_capture
    self.dispatch_captures = dispatch_captures

  def label(self):
    ext = ''
    if self.stack_action == PUSH: ext = '/PUSH(%s)' % self.stack_value
    if self.stack_action == POP: ext = '/POP(%s)' % self.stack_value
    if self.push_retstate: ext += '(call)'
    if self.push_captures: ext += '(cap)'
    if self.dispatch_captures: ext += '(disp)'
    return str(self.token_type) + ext

  def can_combine(self, other):
    return ((self.token_type == EPSILON or other.token_type == EPSILON)
        and (self.stack_action == NONE or other.stack_action == NONE)
        and (self.push_retstate is None or other.push_retstate is None)
        and (not self.dispatch_captures or not other.push_captures)
        and (not self.fixed_capture or not other.fixed_capture)
        )

  def combine(self, other):
    return Transition(
        self.token_type if self.token_type != EPSILON else other.token_type,
        other.target_state,
        first_non(None, self.stack_action, other.stack_action),
        first_non(None, self.stack_value, other.stack_value),
        first_non(None, self.push_captures, other.push_captures),
        first_non(None, self.push_retstate, other.push_retstate),
        first_non(None, self.dispatch_captures, other.dispatch_captures),
        first_non(None, self.capture_token, other.capture_token),
        first_non(None, self.fixed_capture, other.fixed_capture)
        )

  def capture_value(self, value):
    ret = copy.copy(self)
    ret.fixed_capture = value
    return ret

  def __str__(self):
    return '-[ %s ]-> %s' % (self.label(), id(self.target_state))

  __hash__ = hash_all
  __eq__ = eq_all

  def __ne__(self, other):
    return not self == other


def first_non(value, x, y):
  return x if x != value else y


def quoted_string_regex(quote):
  return quote + '(?:[^' + quote + r'\\]|\\.)*' + quote


def make_parser(grammar):
  end_state = EndState()
  start_state = State(grammar.transitions(end_state))
  return ParserFSM(start_state, end_state)


def simplify_fsm(state):
  """Try to remove as many epsilon-transitions as possible, by combining them with
  consuming transitions.

  Actually, this doesn't work very well yet so Imma let it be for now.
  """
  recursion_killer = set()

  def simplify_state(state):
    if state not in recursion_killer:
      recursion_killer.add(state)
      simplify_transitions(state)

  def simplify_transitions(state):
    for t1 in state.transitions():
      for t2 in t1.target_state.transitions():
        if t1.can_combine(t2):
          t1.target_state.remove_transition(t2)
          state.add_transition(t1.combine(t2))

      if not t1.target_state.transitions():
        state.remove_transition(t1)

    for t1 in state.transitions():
      simplify_state(t1.target_state)

  simplify_state(state)
  return state


def top(stack):
  return stack[-1]


class FSMState(object):
  __slots__ = ('state', 'stack', 'return_stack', 'captures')
  def __init__(self, state, stack, return_stack, captures):
    # Reference to current state object
    self.state = state

    # Stack is used for matching pushdown automata (for Forward matching)
    self.stack = stack

    # return_stack is used for returning from routines
    self.return_stack = return_stack

    # Captures are a stack of tuples of capture groups
    self.captures = captures

  def __eq__(self, rhs):
    return self.state == rhs.state and self.stack == rhs.stack and self.captures == rhs.captures

  def __ne__(self, rhs):
    return not self == rhs

  def __hash__(self):
    h = hash(self.state)
    for x in self.stack:
      h = 101 * h + hash(x)
    for x in self.captures:
      h = 101 * h + hash(x)
    for x in self.return_stack:
      h = 101 * h + hash(x)
    return h

  def consume(self, token):
    # Inspect the current state and all states reachable from this state via an
    # epsilon transition.
    return [self.take_transition(t, token)
            for t in self.state.transition_map[token.type]
            if self.can_take_transition(t)]

  def has_epsilons(self):
    return len(self.state.transition_map[EPSILON])

  def epsilons(self):
    # Take all epsilon transitions
    return [self.take_transition(t, EPSILON)
            for t in self.state.transition_map[EPSILON]
            if self.can_take_transition(t)]

  def can_take_transition(self, transition):
    return transition.stack_action != POP or (self.stack and transition.stack_value == top(self.stack))

  def next_stack(self, stack_action, stack_value):
    if stack_action == PUSH:
      return self.stack + (stack_value,)
    elif stack_action == POP:
      return self.stack[:-1]
    else:
      return self.stack

  def next_capture(self, token, capture_token, push, dispatch, fixed_capture):
    new_captures = self.captures[:]
    # Push a new empty capture group
    if push:
      new_captures.append(())
    # Capture the current token into the current group
    if new_captures:
      if fixed_capture: new_captures[-1] = new_captures[-1] + (fixed_capture,)
      if capture_token and token != EPSILON: new_captures[-1] = new_captures[-1] + (token,)
    # Dispatch the current group, append the result to the previous capture group
    if dispatch:
      result = dispatch(*new_captures.pop())
      new_captures[-1] = new_captures[-1] + (result,)
    return new_captures

  def take_transition(self, transition, token):
    new_stack = self.next_stack(transition.stack_action, transition.stack_value)
    new_captures = self.next_capture(token, transition.capture_token, transition.push_captures, transition.dispatch_captures, transition.fixed_capture)
    return_stack = copy.copy(self.return_stack)

    if transition.push_retstate is not None:
      return_stack.append(transition.push_retstate)

    target_state = transition.target_state
    if target_state.returning:
      target_state = return_stack.pop()

    return FSMState(target_state, new_stack, return_stack, new_captures)


class ParserFSM(object):
  def __init__(self, start, end):
    self.start = start
    self.end = end
    self.reset()

  def reset(self):
    captures = [()]
    self.current = set([FSMState(self.start, (), [], captures)])

  def feed(self, token):
    if all(c.state.terminating for c in self.current):
      raise ParseError(token.span, 'Unexpected token past end-of-file: %s' % token.type)

    # Follow all epsilons, then consume, then more epsilons
    states = set(consume_epsilons(self.current))
    states = consume_token(states, token)
    states = set(consume_epsilons(states))

    if not states:
      # Nothing matched--pick any state with actual outgoing transitions to raise an error
      for c in self.current:
        c.state.report_error(token)
      # No state raised an error, make sure we get an error anyway
      raise ParseError(token.span, 'Unexpected %r' % token.type)

    self.current = states

  def finished(self):
    return len(self.current) == 1 and list(self.current)[0].state.terminating

  def parsed_value(self):
    return iter(self.current).next().captures[0]

  def feed_all(self, tokens):
    for token in tokens:
      self.feed(token)

  def assert_finished(self):
    if not self.finished():
      for c in self.current:
        c.state.report_error(Token('end-of-file', '', (0, 0)))


def consume_epsilons(states):
  while states:
    state = states.pop()
    if state.has_epsilons():
      states.update(state.epsilons())
    else:
      yield state


def consume_token(states, token):
  ret = set()
  for state in states:
    ret.update(state.consume(token))
  return ret


class Token(object):
  __slots__ = ('type', 'value', 'span')

  def __init__(self, type, value, span):
    self.type = type
    self.value = value
    self.span = span

  def __repr__(self):
    return 'Token(%r, %r, %r)' % (self.type, self.value, self.span)


def graphvizify(fsm, fobj):
  states = []
  edges = []

  def key(state):
    return 's' + str(id(state))

  def state_label(state):
    return 'x' if state in fsm.current else ''

  def trans_label(trans):
    return trans.label()

  def visit(state):
    if state in states:
      return
    states.append(state)

    fobj.write('  %s [label="%s", peripheries=%d];\n' % (key(state), state_label(state), 2 if state.terminating or
      state.returning else 1))
    for t in state.transitions():
      edges.append((key(state), trans_label(t), key(t.target_state), ''))
      visit(t.target_state)
      if t.push_retstate:
        edges.append((key(state), '(ret)', key(t.push_retstate), ',style="dotted"'))
        visit(t.push_retstate)

  fobj.write('digraph G {\n')

  visit(fsm.end)
  visit(fsm.start)
  for start, label, end, style in edges:
    fobj.write('  %s -> %s [label="%s"%s];\n' % (start, end, label, style))


  fobj.write('}\n')


def quoted_string_regex(quote):
  return quote + '(?:[^' + quote + r'\\]|\\.)*' + quote


def quoted_string_process(value):
  return value[1:-1].decode('string_escape')


class Rule(object):
  def __init__(self, token_type, regex, post_processor=None):
    self.token_type = token_type
    self.regex = regex
    self.post_processor = post_processor


WHITESPACE = Rule('whitespace', r'\s+')
DOUBLE_QUOTED_STRING = Rule('dq_string', quoted_string_regex('"'), quoted_string_process)
SINGLE_QUOTED_STRING = Rule('sq_string', quoted_string_regex("'"), quoted_string_process)
FLOAT = Rule('float', r'-?\d+\.\d+', float)
INTEGER = Rule('integer', r'-?\d+', int)


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
