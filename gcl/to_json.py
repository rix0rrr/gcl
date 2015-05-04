import argparse
import json
import sys

import gcl
from gcl import query
from gcl import util


def main(argv=None, stdin=None):
  parser = argparse.ArgumentParser(description='Convert (parts of) a GCL model file to JSON.')
  parser.add_argument('file', metavar='FILE', type=str, nargs='?',
                      help='File to parse')
  parser.add_argument('selectors', metavar='SELECTOR', type=str, nargs='*',
                      help='Subnodes to convert. The first selector will be treated as the root of the printed output.')

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
    sys.stdout.write(json.dumps(plain))
  except (gcl.ParseError, RuntimeError) as e:
    sys.stderr.write(str(e) + '\n')
    sys.exit(1)
