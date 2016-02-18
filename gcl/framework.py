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


def eval(thunk, env):
  """Evaluate a thunk in an environment.

  Will defer the actual evaluation to the thunk itself, but adds two things:
  caching and recursion detection.

  Since we have to use a global evaluation stack (because there is a variety of functions that may
  be invoked, not just eval() but also __getitem__, and not all of them can pass along a context
  object), GCL evaluation is not thread safe.

  With regard to schemas:

  - A schema can be passed in from outside. The returned object will be validated to see that it
    conforms to the schema. The schema will be attached to the value if possible.
  - Some objects may contain their own schema, such as tuples. This would be out of scope of the
    eval() function, were it not for:
  - Schema validation can be disabled in an evaluation call stack. This is useful if we're
    evaluating a tuple only for its schema information. At that point, we're not interested if the
    object is value-complete.
  """
  key = (thunk.ident, env.ident)
  if key in Activation.stack:
    raise exceptions.EvaluationError('Reference cycle')

  with Activation(key):
    return eval_cache.get(key, lambda: thunk.eval(env))


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

  def exportable_keys(self):
    """Return a list of keys that are exportable from this tuple."""
    pass


# Because we can't trust id(), it'll get reused, we number objects ourselves
# for caching purposes.
obj_nr = 0

def obj_ident():
  global obj_nr
  obj_nr += 1
  return obj_nr


class Activation(object):
  stack = {}
  no_schema_validation = 0

  def __init__(self, key, no_schema_validation=False):
    self.key = key
    self.no_schema_validation = no_schema_validation

  def __enter__(self):
    self.stack[self.key] = self

    # Increase 'no_schema_validation' level
    if self.no_schema_validation:
      Activation.no_schema_validation += 1

  def __exit__(self, value, type, exc):
    del self.stack[self.key]

    # Decrease 'no_schema_validation' level
    if self.no_schema_validation:
      Activation.no_schema_validation -= 1


class Environment(object):
  """Binding environment, inherits from another Environment."""

  def __init__(self, values, parent=None, names=None):
    self.ident = obj_ident()
    self.parent = parent or EmptyEnvironment()
    self.values = values
    self.names = names or values.keys()

  def __getitem__(self, key):
    if key in self.names:
      getter = getattr(self.values, 'get_no_validate', None)
      if getter:
        return getter(key)
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


def make_env(x):
  """Turn an dict into an Environment object."""
  return x if isinstance(x, Environment) else Environment(x)


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


class EnvironmentFunction(object):
  """Wrapper class for a special function that can use the env."""
  def __init__(self, fn):
    self.fn = fn

  def __call__(self, *args, **kwargs):
    return self.fn(*args, **kwargs)


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


