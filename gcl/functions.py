"""GCL standard library functions."""

import math
import itertools
import functools

from os import path

from . import framework


def eager(x):
  """Force eager evaluation of a Thunk, and turn a Tuple into a dict eagerly.

  This forces that there are no unbound variables at parsing time (as opposed
  to later when the variables may be accessed).

  Only eagerly evaluates one level.
  """
  if not hasattr(x, 'items'):
    # The Thunk has been evaluated already, so that was easy :)
    return x

  return dict(x.items())


class StringInterpolationProxy(object):
  """Lazy stringification object from a tuple.

  Also support period-based element access, cuz it looks good.
  """
  def __init__(self, tup, key):
    self.tup = tup
    self.key = key

  def __getattr__(self, key):
    v = self.tup[self.key]
    # If this is a tuple, return another lazy proxy
    if isinstance(v, framework.TupleLike):
      return StringInterpolationProxy(v, key)
    return v[key]

  def __str__(self):
    return str(self.tup[self.key])


def fmt(str, args=None, env=None):
  """String interpolation.

  Normally, we'd just call str.format(**args), but we only want to evaluate
  values from the tuple which are actually used in the string interpolation,
  so we use proxy objects.

  If no args are given, we're able to take the current environment.
  """
  args = args or env
  proxies = {k: StringInterpolationProxy(args, k) for k in args.keys()}
  return str.format(**proxies)


def str_join(lst, sep=' '):
  """Behaves like string.join from Python 2."""
  return sep.join(str(x) for x in lst)


def compose_all(tups):
  """Compose all given tuples together."""
  from . import ast  # I weep for humanity
  return functools.reduce(lambda x, y: x.compose(y), map(ast.make_tuple, tups), ast.make_tuple({}))


def split(string, sep=' '):
  """Splits a string."""
  return string.split(sep)


def has_key(tup, key):
  """Return whether a given tuple has a key and the key is bound."""
  if isinstance(tup, framework.TupleLike):
    return tup.is_bound(key)
  if isinstance(tup, dict):
    return key in tup
  if isinstance(tup, list):
    if not isinstance(key, int):
      raise ValueError('Key must be integer when checking list index')
    return key < len(tup)
  raise ValueError('Not a tuple-like object: %r' % tup)


def flatten(list_of_lists):
  """Flatten a list of lists."""
  ret = []
  for lst in list_of_lists:
    if not isinstance(lst, list):
      raise ValueError('%r is not a list' % lst)
    ret.extend(lst)
  return ret


builtin_functions = {
    'eager': eager,
    'path_join': path.join,
    'join': str_join,
    'fmt': framework.EnvironmentFunction(fmt),
    'sum': sum,
    'compose_all': compose_all,
    'sorted': sorted,
    'split': split,
    'has': has_key,
    'flatten': flatten,
    'sqrt': math.sqrt,  # Only for the Pythagoras demo :)
    'len': len
    }

# Binary operators, by precedence level
binary_operators_before_juxtaposition = [
    {
      '*': lambda x, y: x * y,
      '/': lambda x, y: x / y,
      '%': lambda x, y: x % y,
      'in': lambda x, y: x in y,
    }, {
      '+': lambda x, y: x + y,
      '-': lambda x, y: x - y,
    }]
binary_operators_after_juxtaposition = [
    {
      '==': lambda x, y: x == y,
      '!=': lambda x, y: x != y,
      '<': lambda x, y: x < y,
      '<=': lambda x, y: x <= y,
      '>': lambda x, y: x > y,
      '>=': lambda x, y: x >= y,
    }, {
      # These take thunks
      'and': framework.LazyFunction(lambda x, y: x() and y()),
      'or': framework.LazyFunction(lambda x, y: x() or y()),
    }]
all_binary_operators = {
    k: v
    for os in binary_operators_before_juxtaposition + binary_operators_after_juxtaposition
    for k, v in os.items()}


unary_operators = {
    '-': lambda x: -x,
    'not': lambda x: not x,
    }

