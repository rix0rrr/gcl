"""GCL utility functions."""

import gcl
import hashlib
import sys

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

  def visitError(self, key_path, ex):
    pass

  def visitScalar(self, key_path, value):
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

  keys = sorted(tuple.keys())
  for key in keys:
    key_path = path + [key]
    value = get_or_error(tuple, key)
    if isinstance(value, gcl.Tuple):
      walk(value, walker, key_path)
    elif isinstance(value, Exception):
      walker.visitError(key_path, value)
    else:
      walker.visitScalar(key_path, value)

  walker.leaveTuple(tuple, path)


def select(tuple, selector):
  if hasattr(selector, 'split'):
    selector = selector.split('.')

  for sel in selector:
    tuple = tuple[sel]
  return tuple


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
