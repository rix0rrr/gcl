import argparse
import json
import sys

import gcl
from gcl import query
from gcl import util


def select(dct, path):
  for part in path:
    if not hasattr(dct, 'keys'):
      raise RuntimeError('Value %r cannot be indexed with %r' % (dct, part))
    if part not in dct:
      raise RuntimeError('Value %r has no key %r' % (dct, part))
    dct = dct[part]
  return dct


def main(argv=None, stdin=None):
  parser = argparse.ArgumentParser(description='Convert (parts of) a GCL model file to JSON.')
  parser.add_argument('file', metavar='FILE', type=str, nargs='?',
                      help='File to parse')
  parser.add_argument('selectors', metavar='SELECTOR', type=str, nargs='*',
                      help='Select nodes to include in the JSON.')
  parser.add_argument('--root', '-r', metavar='PATH', type=str, default='',
                      help='Use the indicated root path as the root of the output JSON object (like a.b.c but without wildcards)')

  args = parser.parse_args(argv or sys.argv[1:])

  try:
    if args.file and args.file != '-':
      model = gcl.load(args.file)
    else:
      model = gcl.loads((stdin or sys.stdin).read(), filename='<stdin>')

    sels = query.GPath(args.selectors)
    if not sels.everything():
      model = sels.select(model).deep()

    plain = util.to_python(model)

    selectors = args.root.split('.') if args.root else []
    selected = select(plain, selectors)

    sys.stdout.write(json.dumps(selected, indent=2))
  except (gcl.ParseError, RuntimeError) as e:
    sys.stderr.write(str(e) + '\n')
    sys.exit(1)
