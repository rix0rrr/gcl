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

builtin_functions = {
    'eager': eager,
    'path_join': path.join
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

