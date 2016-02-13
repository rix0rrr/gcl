from . import exceptions

class Thunk(object):
  """An object that is evaluatable in a scope."""
  def eval(self, env):
    raise exceptions.EvaluationError('Not implemented')

  def is_unbound(self):
    """Return whether this thunk represents an actual value (not a placeholder for an unbound value)."""
    return False


class BindableThunk(Thunk):
  """An thunk that is supposed to be bound to a scope before being evaluated.

  We only have objects that are bindable to a tuple's parent environment right now.

  The result of the bind operation should still be a thunk.
  """
  def bind(self, parent_env):
    raise exceptions.EvaluationError('Not implemented')


def eval(thunk, env, schema=None):
  """Evaluate a thunk in an environment.

  Will defer the actual evaluation to the thunk itself, but adds two things:
  caching and recursion detection.

  Since we have to use a global evaluation stack, GCL evaluation is not thread
  safe.
  """
  key = (thunk.ident, env.ident)
  if key in activation_stack:
    raise exceptions.EvaluationError('Reference cycle')

  with Activation(activation_stack, key):
    val = eval_cache.get(key, lambda: thunk.eval(env))
    if schema:
      schema.validate(val)
      setter = getattr(val, 'add_schema', None)
      if setter:
        setter(schema)
    return val


class TupleLike(object):
  """Interface for tuple-like objects."""
  def __getitem__(self, key):
    pass

  def keys(self):
    pass

  def items(self):
    pass

  def __contains__(self):
    pass

  def __iter__(self):
    pass

  def is_bound(self, name):
    """Return whether the value exists and is bound."""
    pass

  def has_key(self, name):
    return name in self


# Because we can't trust id(), it'll get reused, we number objects ourselves
# for caching purposes.
obj_nr = 0

def obj_ident():
  global obj_nr
  obj_nr += 1
  return obj_nr



class Activation(object):
  def __init__(self, stack, key):
    self.stack = stack
    self.key = key

  def __enter__(self):
    self.stack[self.key] = self

  def __exit__(self, value, type, exc):
    del self.stack[self.key]


class Environment(object):
  """Binding environment, inherits from another Environment."""

  def __init__(self, values, parent=None, names=None):
    self.ident = obj_ident()
    self.parent = parent or EmptyEnvironment()
    self.values = values
    self.names = names or values.keys()

  def __getitem__(self, key):
    if key in self.names:
      return self.values[key]
    return self.parent[key]

  def __contains__(self, key):
    if key in self.names:
      return True
    return key in self.parent

  @property
  def root(self):
    if isinstance(self.parent, EmptyEnvironment):
      return self
    else:
      return self.parent.root

  def extend(self, d):
    return Environment(d or {}, self)

  def keys(self):
    return set(self.names).union(self.parent.keys())

  def __repr__(self):
    return 'Environment(%s :: %r)' % (', '.join(self.names), self.parent)


class EmptyEnvironment(object):
  def __init__(self):
    self.ident = obj_ident()

  def __getitem__(self, key):
    raise exceptions.EvaluationError('Unbound variable: %r' % key)

  def __contains__(self, key):
    return False

  def __repr__(self):
    return '<empty>'

  def keys(self):
    return set()

  @property
  def root(self):
    return self

  def extend(self, d):
    return Environment(d or {}, self)


class Cache(object):
  def __init__(self):
    self._cache = {}

  def get(self, key, thunk):
    if key not in self._cache:
      self._cache[key] = thunk()
    return self._cache[key]


eval_cache = Cache()
activation_stack = {}


def is_tuple(x):
  return hasattr(x, 'keys')


def is_list(x):
  from . import ast  # I feel dirty
  return isinstance(x, (list, ast.List))


# Python 2 and 3 compatible string check
try:
    isinstance("", basestring)
    def is_str(s):
        return isinstance(s, basestring)
except NameError:
    def is_str(s):
        return isinstance(s, str)


