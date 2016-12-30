"""GCL query module.

This is a bit haphazard for now, but it defines GPath (yes, very creative),
which supports selection of elements in an object tree.
"""
import functools
import itertools

import pyparsing as p

import gcl
from gcl import framework
from gcl import ast
from gcl import util


class QueryError(RuntimeError):
  pass

def mkAlternates(x):
  return tuple(x)


gcl_grammar = ast.normal_grammar()

variable = gcl_grammar.variable.copy().setParseAction(lambda x: str(x[0]))
alts = ast.bracketedList('{', '}', ',', variable).setParseAction(mkAlternates)
# integer parses as [start_offset, 'matched', end_offset]
list_index = ast.sym('[') + gcl_grammar.integer.copy().setParseAction(lambda ts: int(ts[1])) + ast.sym(']')
everything = p.Literal('*')
element = variable | alts | list_index | everything

selector = p.Group(p.Optional(element + p.ZeroOrMore(ast.sym('.') + element)))


def parseSelector(s):
  try:
    return selector.parseString(s, parseAll=True)[0]
  except p.ParseException as e:
    raise RuntimeError('Error parsing %r: %s' % (s, e))


def listKey(ix):
  return '[%d]' % ix


def isListKey(key):
  return key.startswith('[')


def listKeyIndex(key):
  return int(key[1:-1])


def is_tuple(x):
  return isinstance(x, framework.TupleLike) or isinstance(x, dict)


def negate(fn):
  return lambda x: not fn(x)


def partition(pred, iterable):
    'Use a predicate to partition entries into false entries and true entries'
    # partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9
    t1, t2 = itertools.tee(iterable)
    return list(filter(negate(pred), t1)), list(filter(pred, t2))


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
        if not is_tuple(value) and not isinstance(value, list):
          return

        qhead, qtail = remaining[0], remaining[1:]
        if isinstance(qhead, tuple) and is_tuple(value):
          for alt in qhead:
            if alt in value:
              doSelect(value[alt], pre + [alt], qtail)
        elif qhead == '*':
          if isinstance(value, list):
            indices = range(len(value))
            reprs = [listKey(i) for i in indices]
          else:
            indices = value.keys()
            reprs = indices

          for key, rep in zip(indices, reprs):
            doSelect(value[key], pre + [rep], qtail)
        elif isinstance(qhead, int) and isinstance(value, list):
          doSelect(value[qhead], pre + [listKey(qhead)], qtail)
        elif is_tuple(value):
          if qhead in value:
            doSelect(value[qhead], pre + [qhead], qtail)

    for selector in self.selectors:
      doSelect(model, [], selector)

    return QueryResult(res)


class QueryResult(object):
  def __init__(self, results):
    self.results = results
    self.lists = {}

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
    self.lists = {}
    ret = {}
    for path, value in self.paths_values():
      self.recursiveSet(ret, path, value)
    self.removeMissingValuesFromLists()
    return ret

  def recursiveSet(self, record, path, value):
    # Can't do recursive base case here
    assert len(path) > 0
    head, tail = path[0], path[1:]
    if not tail:
      # Assign value here
      self.ldSet(record, head, value)
    else:
      # Get the element here, recurse. If it doesn't exist yet, create it
      # based on the type of the next path key.
      if not self.ldContains(record, head):
        self.ldSet(record, head, [] if isListKey(tail[0]) else {})

      self.recursiveSet(self.ldGet(record, head), tail, value)

  def ldSet(self, what, key, value):
    """List/dictionary-aware set."""
    if isListKey(key):
      # Make sure we keep the indexes consistent, insert missing_values
      # as necessary. We do remember the lists, so that we can remove
      # missing values after inserting all values from all selectors.
      self.lists[id(what)] = what
      ix = listKeyIndex(key)
      while len(what) <= ix:
        what.append(missing_value)
      what[ix] = value
    else:
      what[key] = value
    return value

  def ldGet(self, what, key):
    """List-aware get."""
    if isListKey(key):
      return what[listKeyIndex(key)]
    else:
      return what[key]

  def ldContains(self, what, key):
    """List/dictinary/missing-aware contains.

    If the value is a "missing_value", we'll treat it as non-existent
    so it will be overwritten by an empty list/dict when necessary to
    assign child keys.
    """
    if isListKey(key):
      i = listKeyIndex(key)
      return i < len(what) and what[i] != missing_value
    else:
      return key in what and what[key] != missing_value

  def removeMissingValuesFromLists(self):
    for lst in self.lists.values():
      i = 0
      while i < len(lst):
        if lst[i] == missing_value:
          lst.pop(i)
        else:
          i += 1

class MissingValue(object):
  pass

missing_value = MissingValue()


class Node(object):
  def __init__(self, path, value):
    self.path = path
    self.value = value

  def id(self):
    return id(self.value)

  def key_name(self):
    return self.path[-1] if len(self.path) else ''

  def __repr__(self):
    return '%s:%r' % ('.'.join(self.path), self.value)


class HasKeyCondition(object):
  def __init__(self, attribute, search_lists=False):
    self.attribute = attribute
    self.search_lists = search_lists

  def matches(self, node):
    return gcl.is_tuple(node.value) and self.attribute in node.value

  def look_in_list(self, path):
    return self.search_lists


class InListCondition(object):
  def __init__(self, list_key):
    self.list_key = list_key

  def matches(self, node):
    return gcl.is_list(node.value) and node.key_name() == self.list_key

  def look_in_list(self, path):
    return False


class TupleFinder(util.ExpressionWalker):
  """Finds tuples in a tree according to a criterion.

  Criteria can be the presence of a specific key, while the object's key itself
  does not starting with a capital letter, or the objects being in a list of a
  key with a particular name.

  If objects refer to one another, they will be sorted in topological order.
  """
  def __init__(self, condition):
    self.condition = condition
    self.ordered = []
    self.unordered = []

  def find(self, root):
    util.walk(root, self)

  def _ids_to_nodes(self, dep_map):
    id_map = {x.id(): x for x in self.unordered}
    return {id_map[k]: [id_map[v] for v in vs] for k, vs in dep_map.items()}

  def order(self):
    # Walk the objects to find the dependencies (by object ID)
    values = [n.value for n in self.unordered]
    dep_finder = DependencyFinder(values)
    util.walk(values, dep_finder)
    self.deps = self._ids_to_nodes(dep_finder.dependencies())

    def has_deps(node):
      return self.deps[node]

    # Now sort baby; iterate through this list multiple times.
    self.ordered = []
    while True:
      next_tranche, self.unordered = partition(has_deps, self.unordered)
      if not next_tranche:
          # Either done or recursive dependencies
          break
      self.ordered.extend(next_tranche)
      # Get rid of dependencies
      for key, list in self.deps.items():
        list[:] = [i for i in list if i not in next_tranche]

  def has_recursive_dependency(self):
    return self.unordered

  def find_recursive_dependency(self):
    """Return a list of nodes that have a recursive dependency."""
    nodes_on_path = []

    def helper(nodes):
      for node in nodes:
        cycle = node in nodes_on_path
        nodes_on_path.append(node)
        if cycle or helper(self.deps.get(node, [])):
          return True
        nodes_on_path.pop()
      return False

    helper(self.unordered)
    return nodes_on_path

  # ExpressionWalker interface
  def enterTuple(self, tuple, path):
    """Called for every tuple.

    If this returns False, the elements of the tuple will not be recursed over
    and leaveTuple() will not be called.
    """
    if skip_name(path):
      return False
    node = Node(path, tuple)
    if self.condition.matches(node):
      self.unordered.append(node)
      return False
    return True

  def enterList(self, xs, path):
    if skip_name(path):
      return False
    node = Node(path, xs)
    if self.condition.matches(node):
      self.unordered.extend([Node(path + ['[%d]' % i], x) for i, x in enumerate(xs)])
      return False
    return self.condition.look_in_list(path)

  def found_one(self, node):
    self.results.append(node)


class DependencyFinder(util.ExpressionWalker):
  """A walker that is used to find dependencies between objects.

  This mechanism totally depends on return value caching, and that we'll get the
  same object back every time.
  """
  def __init__(self, possible_values):
    self.depends_on_ids = {id(x): [] for x in possible_values}
    self.current = None

  def dependencies(self):
    return dict(self.depends_on_ids)

  def enterTuple(self, tuple, path):
    ident = id(tuple)
    if len(path) == 1:
      self.current = ident
    else:
      if ident in self.depends_on_ids:
        self.depends_on_ids[self.current].append(ident)
    return True


def skip_name(path):
  return len(path) and path[-1][0].isupper()

