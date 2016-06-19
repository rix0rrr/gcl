#!/usr/bin/env python
"""Generate documentation for a GCL file."""
import argparse
import collections
import os
import logging
from os import path
import re

import gcl

logger = logging.getLogger('gcl-doc')


class TextFile(object):
  def __init__(self, fobj):
    self.fobj = fobj

  def lines(self, lines):
    for line in lines:
      self.fobj.write(line + '\n')

  def rst_block(self, header, content_lines):
    self.lines(['.. ' + header, ''])
    self.lines(['   %s' % line for line in content_lines])
    self.lines([''])

  def caption(self, title, char='='):
    self.lines([title, char * len(title), ''])


class RstTable(object):
  def __init__(self):
    self.rows = []
    self.w = 0
    self.width = collections.defaultdict(lambda: 1)

  def add_row(self, row):
    self.w = max(self.w, len(row))
    for i, x in enumerate(row):
      self.width[i] = max(self.width[i], len(x))
    self.rows.append(row)

  def generate(self):
    """Generate a list of strings representing the table in RST format."""
    header = ' '.join('=' * self.width[i] for i in range(self.w))
    lines = [
        ' '.join(row[i].ljust(self.width[i]) for i in range(self.w))
        for row in self.rows]
    return [header] + lines + [header]


def slugify(s):
  s = re.sub('[^a-zA-Z0-9-]', '_', s)
  return s if not s.startswith('_') else s[1:]


class Documentation(object):
  def __init__(self, args):
    self.args = args
    self.tuples = {}

  def add_file(self, filename, tuple):
    self.tuples[filename] = DocumentedFile(self.args, filename, tuple)

  def write_dir(self, root):
    for filename, tup in self.tuples.items():
      fullpath = path.join(root, slugify(filename) + '.rst')
      logger.info('Writing %s', fullpath)
      with open(fullpath, 'w') as output_file:
        tup.write(TextFile(output_file))

    if not self.args.no_index:
      fullpath = path.join(root, 'index.rst')
      logger.info('Writing %s', fullpath)
      with open(fullpath, 'w') as output_file:
        self.write_index(TextFile(output_file))

  def write_index(self, fobj):
    fobj.caption(self.args.title)

    fobj.lines([
      '.. toctree::',
      '   :maxdepth: 2',
      ''])
    fobj.lines(['   %s' % slugify(filename) for filename in self.tuples.keys()])


class DocumentedFile(object):
  def __init__(self, args, name, tuple):
    self.args = args
    self.name = name
    self.tuple = tuple
    self.tdoc = TupleDocs(self.args, self.tuple)

  def member_label(self, name):
    return '%s-%s' % (slugify(self.name), name)

  def member_ref(self, name):
    return ':ref:`%s <%s>`' % (name, self.member_label(name))

  def write(self, fobj):
    self.write_title(fobj)
    self.write_contents(fobj)
    self.write_tuple(fobj, self.tuple, self.tdoc, expand_tuples=True)

  def write_title(self, fobj):
    fobj.caption(self.name)

  def write_contents(self, fobj):
    table = RstTable()
    for m, m_node in self.tdoc.tuples():
      table.add_row([self.member_ref(m), m_node.comment.title()])

    if table.generate():
      fobj.rst_block('table::', table.generate())

  def write_tuple(self, fobj, tuple, tdoc, expand_tuples):
    for m, m_node in tdoc.unbound():
      self.write_field(fobj, m, m_node, is_unbound=True)
    for m, m_node in tdoc.scalars():
      self.write_field(fobj, m, m_node, show_default=not m_node.comment.has_tag('hidevalue'))
    for m, m_node in tdoc.tuples():
      self.write_field(fobj, m, m_node, as_section=expand_tuples, show_default=m_node.comment.has_tag('showvalue'))

      if expand_tuples:
        try:
          deeper_tuple = tuple[m]
          self.write_tuple(fobj, deeper_tuple, TupleDocs(self.args, deeper_tuple), expand_tuples=False)
        except Exception as e:
          fobj.rst_block('admonition:: Whoopsie', [str(e)])

  def write_field(self, fobj, name, member_node, is_unbound=False, as_section=False, show_default=False):
    is_input = '(input)' if is_unbound else ''

    comment_text = stylize_comment_block(member_node.comment.lines)
    if as_section:
      fobj.rst_block('_%s:' % self.member_label(name), [])
      fobj.caption(name, '-')
      fobj.lines(comment_text)
      fobj.lines([''])
    else:
      fobj.rst_block('object:: %s %s' % (name, is_input), comment_text)
    if show_default:
      fobj.lines(['   ``Default: %s``' % member_node.value])
    fobj.lines([''])

class TupleDocs(object):
  """Documentation about a single tuple value."""
  def __init__(self, args, tuple):
    self.args = args
    self.tuple = tuple

    members = self._public_members()
    tup, nontup = partition(self._is_tuple, members)
    unbound, scalars = partition(self._is_unbound, nontup)

    self.unbound_members = self._with_nodes(usorted(unbound))
    self.scalar_members = self._with_nodes(usorted(scalars))
    self.tuple_members = self._with_nodes(usorted(tup))

  def _public_members(self):
    return [m for m in self.tuple.keys() if self._is_public_member(m)]

  def _is_public_member(self, name):
    member_comment = self.tuple.get_member_node(name).comment
    return (not self.args.only_documented or member_comment.lines) and member_comment.tag('detail') is None

  def _is_unbound(self, name):
    return not self.tuple.is_bound(name)

  def _is_tuple(self, name):
    try:
      return gcl.is_tuple(self.tuple[name])
    except Exception:
      return False

  def _with_nodes(self, names):
    return [(n, self.tuple.get_member_node(n)) for n in names]

  def unbound(self):
    return self.unbound_members

  def scalars(self):
    return self.scalar_members

  def tuples(self):
    return self.tuple_members

  def all(self):
    return self.unbound() + self.scalars() + self.tuples()


def stylize_comment_block(lines):
  """Parse comment lines and make subsequent indented lines into a code block
  block.
  """
  normal, sep, in_code = range(3)
  state = normal
  for line in lines:
    indented = line.startswith('    ')
    empty_line = line.strip() == ''

    if state == normal and empty_line:
      state = sep
    elif state in [sep, normal] and indented:
      yield ''
      if indented:
        yield '.. code-block:: javascript'
        yield ''
        yield line
        state = in_code
      else:
        state = normal
    elif state == sep and not empty_line:
      yield ''
      yield line
      state = normal
    else:
      yield line
      if state == in_code and not (indented or empty_line):
        sep = normal


def usorted(xs):
  return sorted(xs, key=lambda x: x.lower())


def sort_members(tup, names):
  """Return two pairs of members, scalar and tuple members.

  The scalars will be sorted s.t. the unbound members are at the top.
  """
  scalars, tuples = partition(lambda x: not is_tuple_node(tup.member[x].value), names)
  unbound, bound = partition(lambda x: tup.member[x].value.is_unbound(), scalars)
  return usorted(unbound) + usorted(bound), usorted(tuples)


def public_members(tup, args):
  return [name for name, member in tup.member.items() if is_public(member, args)]


def is_public(member, args):
  return (not args.only_documented or member.comment.lines) and member.comment.tag('detail') is None


def is_tuple_node(x):
  return isinstance(x, gcl.ast.TupleNode)


def partition(pred, xs):
  yes = []
  no = []
  for x in xs:
    if pred(x):
      yes.append(x)
    else:
      no.append(x)
  return yes, no


def resolve_file(fname, paths):
  """Resolve filename relatively against one of the given paths, if possible."""
  fpath = path.abspath(fname)
  for p in paths:
    spath = path.abspath(p)
    if fpath.startswith(spath):
      return fpath[len(spath) + 1:]
  return fname


def main():
  logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

  parser = argparse.ArgumentParser(description='Generate sphinx-compatible documentation')
  parser.add_argument('--title', '-T', default='Documentation', help='Documentation title')
  parser.add_argument('--only-documented', '-d', action='store_true', default=False, help='Only export documented bindings')
  parser.add_argument('--output-dir', '-o', required=True, help='Directory to write output files')
  parser.add_argument('--no-index', '-I', action='store_true', default=False, help='Skip Sphinx index file')
  parser.add_argument('--search-path', '-s', action='append', default=[], help='Resolve GCL files against these paths')
  parser.add_argument('file', metavar='FILE', nargs='+', help='Files to generate documentation for')

  args = parser.parse_args()

  doc = Documentation(args)
  for f in args.file:
    logger.info('Loading %s', f)
    node = gcl.load(f)
    relative_name = resolve_file(f, args.search_path)
    logger.info('Adding as %s', relative_name)
    doc.add_file(relative_name, node)

  if not path.isdir(args.output_dir):
    logger.info('Creating directory %s', args.output_dir)
    os.makedirs(args.output_dir)

  doc.write_dir(args.output_dir)
  logger.info('Done')


if __name__ == '__main__':
  main()
