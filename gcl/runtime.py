"""General runtime classes.

(Implementations of the protocols in gcl.framework).
"""
import collections
import functools
from os import path

from . import framework
from . import schema
from . import exceptions


class Tuple(framework.TupleLike):
  """Bound tuple, with lazy evaluation.

  Contains real values or Thunks. Thunks will be evaluated upon request, but
  not before.

  The parent_env is the environment in which we do lookups for values that are
  not in this Tuple (the lexically enclosing scope).
  """
  def __init__(self, tuplenode, parent_env, dict2tuple):
    self.ident = framework.obj_ident()
    self._ast_node = tuplenode
    self.__parent_env = parent_env
    # This mapping is for backwards compatibility. In principle, the TupleNode owns the members
    self.__items = {m.name: m.value for m in tuplenode.members}
    self.__env_cache = framework.Cache()  # Env cache so eval caching works more effectively

    # This function is only here to break a cyclic dependency on the 'ast' module
    self.dict2tuple = dict2tuple

    self._init_tuple()

  def _init_tuple(self):
    """Initialize shared tuple classes."""
    self.tuple_schema = schema.AnySchema()
    self._value_cache = {}

  def dict(self):
    return self.__items

  def get(self, key, default=None):
    if key in self:
      return self[key]
    return default

  def __getitem__(self, key):
    if type(key) == int:
      raise ValueError('Trying to access tuple as a list')

    if key not in self._value_cache:
      x = self.get_no_validate(key)
      sub_schema = self.tuple_schema.get_subschema(key)
      schema.validate(x, sub_schema)
      schema.attach(x, sub_schema)
      self._value_cache[key] = x

    return self._value_cache[key]

  def get_no_validate(self, key):
    """Return an item without validating the schema."""
    x, env = self.get_thunk_env(key)

    # Check if this is a Thunk that needs to be lazily evaluated before we
    # return it.
    if isinstance(x, framework.Thunk):
      x = framework.eval(x, env)

    return x

  def __contains__(self, key):
    return key in self.__items

  def env(self, current_scope):
    """Return an environment that will look up in current_scope for keys in
    this tuple, and the parent env otherwise.
    """
    return self.__env_cache.get(
            current_scope.ident,
            framework.Environment, current_scope, self.__parent_env, names=self.keys())

  def keys(self):
    return self.__items.keys()

  @property
  def tuples(self):
    return [self]

  def items(self):
    return list(self.iteritems())

  def iteritems(self):
    for k in self.keys():
      yield k, self[k]

  def get_thunk(self, k):
    return self.get_thunk_env(k)[0]

  def get_thunk_env(self, k):
    """Return the thunk AND environment for validating it in for the given key.

    There might be different envs in case the thunk comes from a different (composed) tuple. If the thunk needs its
    environment bound on retrieval, that will be done here.
    """
    if k not in self.__items:
      raise exceptions.EvaluationError('Unknown key: %r in tuple %r' % (k, self))
    x = self.__items[k]

    env = self.env(self)

    # Bind this to the tuple's parent environment
    if isinstance(x, framework.BindableThunk):
      return x.bind(self.__parent_env), env
    return x, env

  def _render(self, key):
    if key in self:
      return '%s = %r' % (key, self.get_thunk(key))
    else:
      return '%s' % key

  def is_bound(self, name):
    return name in self and not self.get_thunk(name).is_unbound()

  def get_member_node(self, name):
    """Return the AST node for the given member."""
    return self._ast_node.member[name]

  def compose(self, tup):
    if not isinstance(tup, Tuple):
      tup = self.dict2tuple(tup)

    composite = CompositeTuple(self.tuples + tup.tuples, self.dict2tuple)
    composite.attach_schema(self.tuple_schema)
    composite.attach_schema(tup.tuple_schema)
    return composite

  def attach_schema(self, schem):
    """Add a tuple schema to this object (externally imposed)"""
    self.tuple_schema = schema.AndSchema.make(self.tuple_schema, schem)

  def get_schema_spec(self, key):
    """Return the evaluated schema expression from a subkey."""
    member_node = self._ast_node.member.get(key, None)
    if not member_node:
      return schema.AnySchema()

    s = framework.eval(member_node.member_schema, self.env(self))
    if not isinstance(s, schema.Schema):
      raise ValueError('Node %r with schema node %r should evaluate to Schema, got %r' % (member_node, member_node.member_schema, s))
    return s

  def get_required_fields(self):
    """Return the names of fields that are required according to the schema."""
    return [m.name for m in self._ast_node.members if m.member_schema.required]

  def _keys_and_privacy(self):
    return {k: self._ast_node.member[k].member_schema.private for k in self.keys()}

  def exportable_keys(self):
    """Return a list of keys that are exportable from this tuple."""
    return [k for k, v in self._keys_and_privacy().items() if not v]

  def __iter__(self):
    return iter(self.keys())

  def __repr__(self):
    return '{%s}' % '; '.join(self._render(k) for k in self.keys())

  def __call__(self, right):
    """Apply a tuple to another value."""
    if framework.is_tuple(right):
      return self.compose(right)

    if framework.is_str(right):
      return self[right]

    raise exceptions.EvaluationError("Can't apply tuple (%r) to argument (%r): string or tuple expected" % (self, right))


class CompositeBaseTuple(object):
  """A tuple-like object that will be used to resolve 'base' to.

  This will start looking in the tuples of the composite, from right to left,
  and check the complete composite for declared v
  """
  def __init__(self, composite, index):
    self.composite = composite
    self.index = index

  def __getitem__(self, key):
    for tup, env in self.composite.lookups[self.index:]:
      if key in tup:
        # Can't simply use tup[key] because we need to get the indicated thunk but evaluate it in
        # the CURRENT environment. The environment there is a combination of all tuples to the left
        # of the indicated tuple.
        thunk = tup.get_thunk(key)
        if not isinstance(thunk, framework.Thunk):
          return thunk
        if not thunk.is_unbound():
          return framework.eval(thunk, env)
    raise exceptions.EvaluationError('Unknown key: base.%r in composite tuple:' % (key, self.composite))


def env_of(tup, self):
  if isinstance(tup, Tuple):
    return tup.env(self)
  return tup


class CompositeTuple(Tuple):
  """2 or more composited tuples.

  Keys are looked up from right-to-left, and every key will be evaluated in its
  tuple's own environment, except the 'current_scope' will be set to the
  CompositeTuple (so that declared names will be looked up in the composite
  tuple).

  To properly resolve the special variable 'base', we construct smaller
  composite tuples which only contain the tuples to the left of each tuple,
  which will get returned as the result of the expression 'base'.
  """
  def __init__(self, tuples, dict2tuple):
    self.ident = framework.obj_ident()
    self._tuples = tuples
    self._keys = functools.reduce(lambda s, t: s.union(t.keys()), self._tuples, set())
    self._makeLookupList()
    self.dict2tuple = dict2tuple

    self._init_tuple()

  def _makeLookupList(self):
    # Count index from the back because we're going to reverse
    envs = [framework.Environment({'base': CompositeBaseTuple(self, len(self.tuples) - i)}, env_of(t, self)) for i, t in enumerate(self.tuples)]
    self.lookups = list(zip(self._tuples, envs))
    self.lookups.reverse()

  @property
  def tuples(self):
    return self._tuples

  def __contains__(self, key):
    return key in self._keys

  def keys(self):
    return list(self._keys)

  def items(self):
    return [(k, self[k]) for k in self.keys()]

  def get(self, key, default=None):
    if key in self:
      return self[key]
    return default

  def is_bound(self, name):
    return name in self and self.maybe_get_thunk_env(name)[0] is not None

  def maybe_get_thunk_env(self, key):
    for tup, env in self.lookups:
      if key in tup:
        thunk = tup.get_thunk(key)
        if not isinstance(thunk, framework.Thunk) or not thunk.is_unbound():
          return (thunk, env)
    return None, None

  def get_thunk_env(self, key):
    thunk, env = self.maybe_get_thunk_env(key)
    if thunk:
      return thunk, env
    raise exceptions.EvaluationError('Unknown key: %r in composite tuple %r' % (key, self))

  def get_member_node(self, key):
    """Return the AST node for the given member, from the first tuple that serves it."""
    for tup, _ in self.lookups:
      if key in tup:
        return tup.get_member_node(key)
    raise RuntimeError('Key not found in composite tuple: %r' % key)

  def exportable_keys(self):
    """Return a list of keys that are exportable from this tuple.

    Returns all keys that are not private in any of the tuples.
    """
    keys = collections.defaultdict(list)
    for tup in self._tuples:
      for key, private in tup._keys_and_privacy().items():
        keys[key].append(private)
    return [k for k, ps in keys.items() if not any(ps)]

  def __repr__(self):
    return ' '.join(repr(t) for t in self.tuples)


#----------------------------------------------------------------------

class OnDiskFiles(object):
  """Abstraction of a file system, with search path."""
  def __init__(self, search_path=[]):
    self.search_path = search_path

  def resolve(self, current_file, rel_path):
    """Search the filesystem."""
    search_path = [path.dirname(current_file)] + self.search_path

    target_path = None
    for search in search_path:
      if path.isfile(path.join(search, rel_path)):
        target_path = path.normpath(path.join(search, rel_path))
        break

    if not target_path:
      raise exceptions.EvaluationError('No such file: %r, searched %s' %
                            (rel_path, ':'.join(search_path)))

    return target_path, path.abspath(target_path)

  def load(self, path):
    with open(path, 'r') as f:
      return f.read()


class InMemoryFiles(object):
  """Simulate a filesystem from an in-memory dictionary.

  The dictionary maps path to file contents.
  """
  def __init__(self, file_dict):
    self.file_dict = file_dict

  def resolve(self, current_file, rel_path):
    """Search the filesystem."""
    p = path.join(path.dirname(current_file), rel_path)
    if p not in self.file_dict:
      raise RuntimeError('No such fake file: %r' % p)
    return p, p

  def load(self, path):
    return self.file_dict[path]


