"""Script to print a GCL model."""

import argparse
import sys

import gcl
from gcl import util

class PrintWalker(util.ExpressionWalker):
  def __init__(self, table, prefix_cells=[], lowercase=False, errors_only=False):
    self.table = table
    self.prefix_cells = prefix_cells
    self.do_prefix = False
    self.lowercase = lowercase
    self.errors_only = errors_only

  def _printableKey(self, k):
    return not self.lowercase or ('a' <= k[0] <= 'z')

  def _newLine(self):
    self.table.feedLine()
    self.do_prefix = True

  def _maybePrintPrefix(self):
    if self.do_prefix:
      self.table.addAll(self.prefix_cells)
      self.do_prefix = False

  def _outline(self, path):
    return util.Cell('|  ' * len(path), 'cyan')

  def _printBullet(self, path):
    self._maybePrintPrefix()
    self.table.add(self._outline(path[:-1])
                       .write('+- ', 'cyan')
                       .write(path[-1]))

  def _printArrow(self):
    self.table.add(util.Cell('=>', 'cyan'))

  def _printRecordSeparator(self):
    self._maybePrintPrefix()
    self.table.add(util.Cell('----------', 'yellow'))
    self._newLine()

  def enterTuple(self, tuple, path):
    if path:
      if not self._printableKey(path[-1]):
        return False
      self._printBullet(path)
      self._newLine()

    return True

  def leaveTuple(self, tuple, path):
    pass

  def visitError(self, key_path, ex):
    self._printBullet(key_path)
    self._printArrow()
    self.table.add(util.Cell('<%s>' % ex, 'red'))
    self._newLine()

  def _printList(self, key_path, xs):
    if not len(xs):
      self.table.add(util.Cell('[]', 'yellow'))
      self._newLine()

    prefix = self.prefix_cells + [self._outline(key_path), util.Cell()]
    w2 = PrintWalker(self.table, prefix, lowercase=self.lowercase, errors_only=self.errors_only)
    for i, x in enumerate(xs):
      if isinstance(x, gcl.Tuple):
        if i:
          w2._printRecordSeparator()
        util.walk(x, w2)
      else:
        w2._printScalar([], x)

    if not w2.do_prefix:
      self._newLine()

  def _printScalar(self, key_path, value):
    self._maybePrintPrefix()
    if isinstance(value, list):
      self._printList(key_path, value)
    else:
      self.table.add(util.Cell(repr(value)))
      self._newLine()

  def visitScalar(self, key_path, value):
    if self.errors_only:
      return

    if not self._printableKey(key_path[-1]):
      return

    self._printBullet(key_path)
    self._printArrow()
    self._printScalar(key_path, value)


def print_model(model, **kwargs):
  table = util.ConsoleTable()
  util.walk(model, PrintWalker(table, **kwargs))
  table.printOut(sys.stdout)

def print_selectors(model, selectors, **kwargs):
  if not selectors:
    print_model(model, **kwargs)
  else:
    for selector in selectors:
      try:
        print(util.Color.colorize(selector, 'yellow'))
        print_model(util.select(model, selector), **kwargs)
        print('')
      except RuntimeError as e:
        print(util.Color.colorize(str(e), 'red'))


def main(argv=None, stdin=None):
  parser = argparse.ArgumentParser(description='Print a GCL model file.')
  parser.add_argument('file', metavar='FILE', type=str, nargs='?',
                      help='File to parse')
  parser.add_argument('selectors', metavar='SELECTOR', type=str, nargs='*',
                      help='Subnodes to print')
  parser.add_argument('-e', '--errors-only', dest='errors_only', action='store_true', default=False,
                      help='Only show errors')
  parser.add_argument('-l', '--lowercase', dest='lowercase', action='store_true', default=False,
                      help='Don\'t recurse into variables starting with capital letters.')

  args = parser.parse_args(argv or sys.argv[1:])

  try:
    if args.file and args.file != '-':
      model = gcl.load(args.file)
    else:
      model = gcl.loads((stdin or sys.stdin).read(), filename='<stdin>')
  except gcl.ParseError as e:
    print(e)
  else:
    print_selectors(model, args.selectors, lowercase=args.lowercase, errors_only=args.errors_only)
