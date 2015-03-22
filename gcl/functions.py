"""GCL standard library functions."""

from os import path
import gcl


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
    if isinstance(v, gcl.Tuple):
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


class EnvironmentFunction(object):
  """Wrapper class for a special function that can use the env."""
  def __init__(self, fn):
    self.fn = fn

  def __call__(self, *args, **kwargs):
    return self.fn(*args, **kwargs)


builtin_functions = {
    'eager': eager,
    'path_join': path.join,
    'fmt': EnvironmentFunction(fmt)
    }


# Binary operators, by precedence level
binary_operators = [
    {
      '*': lambda x, y: x * y,
      '/': lambda x, y: x / y,
    }, {
      '+': lambda x, y: x + y,
      '-': lambda x, y: x - y,
    }, {
      '==': lambda x, y: x == y,
      '!=': lambda x, y: x != y,
      '<': lambda x, y: x < y,
      '<=': lambda x, y: x <= y,
      '>': lambda x, y: x > y,
      '>=': lambda x, y: x >= y,
    }, {
      'and': lambda x, y: x and y,
      'or': lambda x, y: x or y,
    }]
all_binary_operators = {k: v for os in binary_operators for k, v in os.items()}


unary_operators = {
    '-': lambda x: -x,
    'not': lambda x: not x,
    }

