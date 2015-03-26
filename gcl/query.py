"""GCL query module."""


import pyparsing as p

import gcl


class QueryError(RuntimeError):
  pass

def bla(x):
  return tuple(x)

variable = gcl.variable.copy().setParseAction(lambda x: str(x[0]))
alts = gcl.bracketedList('{', '}', ',', variable, bla)
everything = p.Literal('*')
element = variable | alts | everything

selector = p.Group(p.Optional(element + p.ZeroOrMore(gcl.sym('.') + element)))


def parseSelector(s):
  return selector.parseString(s, parseAll=True)[0]


class GPath(object):
  """Path selector specifications

  Format:

    name1.name2.*.name3.{name5,name6,name7}
  """
  def __init__(self, spec):
    if not isinstance(spec, list):
      spec = [spec]
    self.selectors = [list(parseSelector(x)) for x in spec]

  def everything(self):
    return not self.selectors

  def select(self, model):
    """Select nodes according to the input selector.

    This can ALWAYS return multiple root elements.
    """
    res = []

    def doSelect(value, pre, remaining):
      if not remaining:
        res.append((pre, value))
      else:
        # For the other selectors to work, value must be a Tuple or a list at this point.
        if not isinstance(value, gcl.Tuple) and not isinstance(value, list):
          return

        qhead, qtail = remaining[0], remaining[1:]
        if isinstance(qhead, tuple):
          for alt in qhead:
            if alt in value:
              doSelect(value[alt], pre + [alt], qtail)
        elif qhead == '*':
          if isinstance(value, gcl.Tuple):
            indices = value.keys()
            reprs = indices
          else:
            indices = range(len(value))
            reprs = ['[%d]' % i for i in indices]

          for key, rep in zip(indices, reprs):
            doSelect(value[key], pre + [rep], qtail)
        else:
          if qhead in value:
            doSelect(value[qhead], pre + [qhead], qtail)

    for selector in self.selectors:
      doSelect(model, [], selector)

    return QueryResult(res)


class QueryResult(object):
  def __init__(self, results):
    self.results = results

  def empty(self):
    return not self.results

  def first(self):
    return self.results[0][1]

  def values(self):
    for path, value in self.results:
      yield value

  def paths_values(self):
    return self.results

  def deep(self):
    """Return a deep dict of the values selected.

    The leaf values may still be gcl Tuples. Use util.to_python() if you want
    to reify everything to real Python values.
    """

    def set(what, key, value):
      """List-aware set"""
      if key.startswith('['):
        what.append(value)
      else:
        what[key] = value
      return value

    def get(what, key):
      """List-aware get"""
      if key.startswith('['):
        return what[int(key[1:-1])]
      else:
        return what[key]

    def contains(what, key):
      """List-aware contains."""
      if key.startswith('['):
        return int(key[1:-1]) < len(what)
      else:
        return key in what

    ret = {}
    for path, value in self.paths_values():
      d = ret
      for i, part in enumerate(path[:-1]):
        if not contains(d, part):
          if path[i+1].startswith('['):
            d = set(d, part, [])
          else:
            d = set(d, part, {})
        else:
          d = get(d, part)
      set(d, path[-1], value)
    return ret
