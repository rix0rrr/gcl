"""GCL utility functions."""

import gcl
import hashlib
import sys


class RecursionException(RuntimeError):
  pass


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

  def visitRecursion(self, key_path):
    self.visitError(key_path, RecursionException('recursion'))

  def visitError(self, key_path, ex):
    pass

  def visitScalar(self, key_path, value):
    pass


def get_or_error(tuple, key):
  try:
    return tuple[key]
  except gcl.EvaluationError as e:
    return e


def to_python(value, seen=None):
  """Reify values to their Python equivalents.

  Does recursion detection, failing when that happens.
  """
  seen = seen or set()
  if isinstance(value, gcl.Tuple):
    if value.ident in seen:
      raise RecursionException('to_python: infinite recursion while evaluating %r' % value)
    seen.add(value.ident)
    return {k: to_python(value[k], seen=seen) for k in value.keys()}
  if isinstance(value, list):
    return [to_python(x, seen=seen) for x in value]
  return value


def walk(value, walker, path=None, seen=None):
  """Walks the _evaluated_ tree of the given GCL tuple.

  The appropriate methods of walker will be invoked for every element in the
  tree.
  """
  seen = seen or set()
  path = path or []

  # Recursion
  if id(value) in seen:
    walker.visitRecursion(path)
    return

  # Error
  if isinstance(value, Exception):
    walker.visitError(path, value)
    return

  # List
  if isinstance(value, list):
    # Not actually a tuple, but okay
    if walker.enterTuple(value, path) is False:
      return
    for i, x in enumerate(value):
      walk(x, walker, path=path + ['[%d]' % i], seen=seen)
    walker.leaveTuple(value, path)
    return

  # Scalar
  if not isinstance(value, gcl.Tuple):
    walker.visitScalar(path, value)
    return

  # Tuple
  if walker.enterTuple(value, path) is False:
    return

  seen.add(id(value))  # Anti-recursion

  keys = sorted(value.keys())
  for key in keys:
    key_path = path + [key]
    elm = get_or_error(value, key)
    walk(elm, walker, path=key_path, seen=seen)

  walker.leaveTuple(value, path)

  seen.remove(id(value))


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


class Color(object):
  yellow = '\033[93m'
  endc = '\033[0m'
  red = '\033[91m'
  green = '\033[92m'
  cyan = '\033[96m'

  @classmethod
  def noColorize(cls, text, color):
    return text

  @classmethod
  def doColorize(cls, text, color):
    if not color:
      return text
    return '%s%s%s' % (getattr(cls, color), text, cls.endc)

  if sys.stdout.isatty():
    colorize = doColorize
  else:
    colorize = noColorize


class Cell(object):
  def __init__(self, text='', color=''):
    self.text = ''
    self.len = 0
    if text:
      self.write(text, color)

  def write(self, text, color=''):
    self.text += Color.colorize(text, color)
    self.len += len(text)
    return self

  def copy(self):
    import copy
    return copy.copy(self)


class ConsoleTable(object):
  """Class to print a table of varying column sizes."""
  def __init__(self):
    # Cells are lists with a text and a width
    self.table = [[]]

  def add(self, cell):
    self.table[-1].append(cell)

  def addAll(self, cells):
    self.table[-1].extend(cells)

  def feedLine(self):
    self.table.append([])

  def currentRowCopy(self):
    return [c.copy() for c in self.table[-1]]

  def _findColumnSizes(self):
    sizes = []
    for row in self.table:
      while len(sizes) < len(row):
        sizes.append(0)
      for i, cell in enumerate(row[:-1]):
        sizes[i] = max(sizes[i], cell.len)
    return sizes

  def printOut(self, fobj):
    sizes = self._findColumnSizes()
    for row in self.table:
      for i, cell in enumerate(row):
        if i != 0:
          fobj.write(' ')
        fobj.write(cell.text)
        if i < len(row) - 1:
          fobj.write(' ' * (sizes[i] - cell.len))
      fobj.write('\n')
