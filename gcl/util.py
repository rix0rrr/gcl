"""GCL utility functions."""

import gcl
import hashlib

class ExpressionWalker(object):
  """Defines the interface for walk()."""

  def enterTuple(self, tuple, path):
    """Called for every tuple.

    If this returns False, the elements of the tuple will not be recursed over
    and leaveTuple() will not be called.
    """
    pass

  def leaveTuple(self, tuple, path):
    pass

  def visitError(self, key, ex, path):
    pass

  def visitScalar(self, key, value, path):
    pass


def get_or_error(tuple, key):
  try:
    return tuple[key]
  except Exception as e:
    return e


def walk(tuple, walker, path=None):
  """Walks the _evaluated_ tree of the given GCL tuple.

  The appropriate methods of walker will be invoked for every element in the
  tree.
  """
  path = path or []
  if not isinstance(tuple, gcl.Tuple):
    raise ValueError('Argument to walk() must be a GCL tuple')

  if walker.enterTuple(tuple, path) is False:
    return  # Do nothing

  keys = tuple.keys()
  for key in keys:
    value = get_or_error(tuple, key)
    if isinstance(value, gcl.Tuple):
      walk(value, walker, path + [key])
    elif isinstance(value, Exception):
      walker.visitError(key, value, path)
    else:
      walker.visitScalar(key, value, path)

  walker.leaveTuple(tuple, path)


def _digest(value, digest):
  if isinstance(value, gcl.Tuple):
    digest.update('T')
    for k in value.keys():
      v = get_or_error(value, k)
      digest.update('K' + k)
      _digest(v, digest)
    digest.update('E')
  elif isinstance(value, list):
    digest.update('L')
    for x in value:
      _digest(x, digest)
    digest.update('E')
  else:
    # Otherwise add the string representation of value to the digest.
    digest.update('S' + str(value))

def fingerprint(value):
  """Return a hash value that uniquely identifies the GCL value."""
  h = hashlib.sha256()
  _digest(value, h)
  return h.digest().encode('hex')
