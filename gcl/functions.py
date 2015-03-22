"""GCL standard library functions."""

from os import path


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


class ValueProxy(object):
  """Lazy stringification object from a tuple."""
  def __init__(self, tup, key):
    self.tup = tup
    self.key = key

  def __str__(self):
    return str(self.tup[self.key])


def fmt(str, args):
  """String interpolation.

  Normally, we'd just call str.format(**args), but we only want to evaluate
  values from the tuple which are actually used in the string interpolation,
  so we use proxy objects.
  """
  proxies = {k: ValueProxy(args, k) for k in args.keys()}
  return str.format(**proxies)


builtin_functions = {
    'eager': eager,
    'path_join': path.join,
    'fmt': fmt
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

