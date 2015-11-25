"""Script to print a GCL model."""

import argparse
import sys

import gcl
from gcl import util
from gcl import query


class QualifiedPrintWalker(util.ExpressionWalker):
  def __init__(self, lowercase=False, errors_only=False):
    self.lowercase = lowercase
    self.errors_only = errors_only

  def _printableKey(self, k):
    return not self.lowercase or ('a' <= k[0] <= 'z')

  def enterTuple(self, tuple, path):
    if path:
      if not self._printableKey(path[-1]):
        return False
    return True

  def leaveTuple(self, tuple, path):
    pass

  def visitError(self, key_path, ex):
    print('%s = <%s>' % ('.'.join(key_path), ex))

  def visitScalar(self, key_path, value):
    if self.errors_only:
      return
    if not self._printableKey(key_path[-1]):
      return

    print('%s = %s' % ('.'.join(key_path), value))


class PrettyPrintWalker(util.ExpressionWalker):
  def __init__(self, table, prefix_cells=[], lowercase=False, errors_only=False):
    self.table = table
    self.prefix_cells = prefix_cells
    self.do_prefix = False
    self.lowercase = lowercase
    self.errors_only = errors_only
    self.tuple_paths = [[]]

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

    self.tuple_paths.append(path)
    return True

  def leaveTuple(self, tuple, path):
    self.tuple_paths.pop()

  def visitError(self, key_path, ex):
    self._printBullet(key_path)
    self._printArrow()
    self.table.add(util.Cell('<%s>' % util.compact_error(ex), 'red'))
    self._newLine()

  def visitScalar(self, key_path, value):
    if self.errors_only:
      return
    if not key_path:
      self.table.add(util.Cell(repr(value)))
      return
    if not self._printableKey(key_path[-1]):
      return

    self._printBullet(key_path)
    self._printArrow()
    self._maybePrintPrefix()
    self.table.add(util.Cell(repr(value)))
    self._newLine()


def pretty_print_model(model, **kwargs):
  table = util.ConsoleTable()
  util.walk(model, PrettyPrintWalker(table, **kwargs))
  table.printOut(sys.stdout)


def qualified_print(model, **kwargs):
  util.walk(model, QualifiedPrintWalker(**kwargs))


def print_selectors(model, selectors, printer, **kwargs):
  sels = query.GPath(selectors)
  if sels.everything():
    printer(model, **kwargs)
  else:
    for path, value in sels.select(model).paths_values():
      try:
        print(util.Color.colorize('.'.join(path), 'yellow'))
        printer(value, **kwargs)
        print('')
      except gcl.EvaluationError as e:
        print(util.Color.colorize(str(e), 'red'))


def main(argv=None, stdin=None):
  parser = argparse.ArgumentParser(description='Print a GCL model file.')
  parser.add_argument('file', metavar='FILE', type=str, nargs='?',
                      help='File to parse')
  parser.add_argument('selectors', metavar='SELECTOR', type=str, nargs='*',
                      help='Subnodes to print')
  parser.add_argument('-e', '--errors-only', dest='errors_only', action='store_true', default=False,
                      help='Only show errors')
  parser.add_argument('-q', '--qualified-paths', dest='qualified_paths', action='store_true', default=False,
                      help='Show qualified paths')
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
    sys.exit(1)
  else:
    printer = qualified_print if args.qualified_paths else pretty_print_model
    print_selectors(model, args.selectors, printer, lowercase=args.lowercase, errors_only=args.errors_only)
