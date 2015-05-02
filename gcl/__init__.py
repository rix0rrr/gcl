"""
GCL -- Generic Configuration Language

See README.md for an explanation of GCL and concepts.
"""

import functools
from os import path

import pyparsing as p

from . import functions

__version__ = '0.4.13'


class GCLError(RuntimeError):
  pass


class ParseError(GCLError):
  pass


class EvaluationError(GCLError):
  def __init__(self, message, inner=None):
    super(EvaluationError, self).__init__(message)
    self.inner = inner

  def __str__(self):
    return self.args[0] + ('\n' + str(self.inner) if self.inner else '')


def do(*fns):
  def fg(args):
    for fn in fns:
      args = fn(args)
    return args
  return fg

def doapply(what):
  def fn(args):
    return what(*args)
  return fn

def head(x):
  return x[0]

def second(x):
  return x[1]

def inner(x):
  return x[1:-1]

def mkBool(s):
  return True if s == 'true' else False

def drop(x):
  return []

def find_relative(current_dir, rel_path):
  if rel_path.startswith('/'):
    return rel_path
  else:
    return path.normpath(path.join(current_dir, rel_path))


class Cache(object):
  def __init__(self):
    self._cache = {}

  def get(self, key, thunk):
    if key not in self._cache:
      self._cache[key] = thunk()
    return self._cache[key]


class Activation(object):
  def __init__(self, stack, key):
    self.stack = stack
    self.key = key

  def __enter__(self):
    self.stack[self.key] = self

  def __exit__(self, value, type, exc):
    del self.stack[self.key]


# Because we can't trust id(), it'll get reused, we number objects ourselves
# for caching purposes.
obj_nr = 0

def obj_ident():
  global obj_nr
  obj_nr += 1
  return obj_nr


eval_cache = Cache()
activation_stack = {}

def eval(thunk, env):
  """Evaluate a thunk in an environment.

  Will defer the actual evaluation to the thunk itself, but adds two things:
  caching and recursion detection.

  Since we have to use a global evaluation stack, GCL evaluation is not thread
  safe.
  """
  #with self._evals.evaluating((id(env), key), key):
  key = (thunk.ident, env.ident)
  if key in activation_stack:
    raise EvaluationError('Reference cycle')
  with Activation(activation_stack, key):
    return eval_cache.get(key, lambda: thunk.eval(env))


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
      raise EvaluationError('No such file: %r, searched %s' %
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


class NormalLoader(object):
  def __init__(self, fs):
    self.fs = fs
    self.cache = Cache()

  def __call__(self, current_file, rel_path):
    nice_path, full_path = self.fs.resolve(current_file, rel_path)

    # Cache on full path, but tell script about nice path
    do_load = lambda: loads(self.fs.load(full_path), filename=nice_path, loader=self)
    return self.cache.get(full_path, do_load)


def loader_with_search_path(search_path):
  return NormalLoader(OnDiskFiles(search_path))


# Default loader doesn't have any search path
default_loader = NormalLoader(OnDiskFiles())

# Python 2 and 3 compatible string check
try:
    isinstance("", basestring)
    def is_str(s):
        return isinstance(s, basestring)
except NameError:
    def is_str(s):
        return isinstance(s, str)


#----------------------------------------------------------------------
#  Model
#

class ParseContext(object):
  def __init__(self):
    self.filename = '<from string>'
    self.loader = None

the_context = ParseContext()


class EmptyEnvironment(object):
  def __init__(self):
    self.ident = obj_ident()

  def __getitem__(self, key):
    raise EvaluationError('Unbound variable: %r' % key)

  def __contains__(self, key):
    return False

  def __repr__(self):
    return '<empty>'

  def keys(self):
    return set()


class SourceLocation(object):
  def __init__(self, string, offset):
    self.filename = the_context.filename
    self.string = string
    self.offset = offset

  @property
  def line(self):
    return p.line(self.offset, self.string)

  @property
  def lineno(self):
    return p.lineno(self.offset, self.string)

  @property
  def col(self):
    return p.col(self.offset, self.string)

  @property
  def line_spec(self):
    return '%s:%s' % (self.filename, self.lineno)

  def error_in_context(self, msg):
    msg = '%s:%d: %s in \'%s\'' % (self.filename, self.lineno, msg, self.line)
    return msg


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

  def extend(self, d):
    return Environment(d or {}, self)

  def keys(self):
    return set(self.names).union(self.parent.keys())

  def __repr__(self):
    return 'Environment(%s :: %r)' % (', '.join(self.names), self.parent)


class Thunk(object):
  def eval(self, env):
    raise EvaluationError('Whoops')


class Null(Thunk):
  """Null, evaluates to None."""
  def __init__(self):
    self.ident = obj_ident()

  def eval(self, env):
    return None

  def __repr__(self):
    return "null";


class Void(Thunk):
  """A missing value."""
  def __init__(self, name, location):
    self.name = name
    self.location = location
    self.ident = obj_ident()

  def eval(self, env):
    raise EvaluationError(self.location.error_in_context('Unbound value: %r' % self.name))

  def __repr__(self):
    return '<unbound>'


class Inherit(Thunk):
  """Inherit Thunks can be either bound or unbound."""

  def __init__(self, name=None, env=None):
    self.ident = obj_ident()
    self.name = name
    self.env = env

  def eval(self, env):
    if not self.env:
      raise EvaluationError("Shouldn't evaluate Inherit nodes")
    return self.env[self.name]

  def __repr__(self):
    return 'inherit %s' % self.name


def mkInherits(tokens):
  return [(t, Inherit()) for t in list(tokens)]


class Constant(Thunk):
  """A GCL constant expression."""
  def __init__(self, value):
    self.ident = obj_ident()
    self.value = value

  def eval(self, env):
    return self.value

  def __repr__(self):
    if type(self.value) == bool:
      return 'true' if self.value else 'false'
    return repr(self.value)


class Var(Thunk):
  """Reference to another value."""
  def __init__(self, name, location):
    self.ident = obj_ident()
    self.name = name
    self.location = location

  def eval(self, env):
    try:
      return env[self.name]
    except EvaluationError as e:
      raise EvaluationError(self.location.error_in_context('while evaluating %r' % self.name), e)

  def __repr__(self):
    return self.name


def mkVar(s, loc, toks):
  return Var(toks[0], SourceLocation(s, loc))


class List(Thunk):
  """A GCL list."""
  def __init__(self, values):
    self.ident = obj_ident()
    self.values = values

  def eval(self, env):
    return [eval(v, env) for v in self.values]

  def __repr__(self):
    return repr(self.values)


class ArgList(Thunk):
  """A paren-separated argument list.

  This is actually a shallow wrapper for Python's list type. We can't use that
  because pyparsing will automatically concatenate lists, which we don't want
  in this case.
  """
  def __init__(self, values):
    self.ident = obj_ident()
    self.values = values

  def eval(self, env):
    return [eval(v, env) for v in self.values]

  def __repr__(self):
    return '(%s)' % ', '.join(repr(x) for x in self.values)


class UnboundTuple(Thunk):
  """Unbound tuple.

  When evaluating, the tuple doesn't actually evaluate its children. Instead,
  we return a (lazy) Tuple object that only evaluates the elements when they're
  requested.
  """
  def __init__(self, kv_pairs):
    self.ident = obj_ident()
    self.items = dict(kv_pairs)
    self._cache = Cache()

  def eval(self, env):
    return self._cache.get(env.ident, lambda: Tuple(self.items, env))

  def __repr__(self):
    return ('{' +
            '; '.join('%s = %r' % (key, value) for key, value in self.items.items()) +
            '}')


class Tuple(object):
  """Bound tuple, with lazy evaluation.

  Contains real values or Thunks. Thunks will be evaluated upon request, but
  not before.

  The parent_env is the environment in which we do lookups for values that are
  not in this Tuple (the lexically enclosing scope).
  """
  def __init__(self, items, parent_env):
    self.ident = obj_ident()
    self.__items = items
    self._parent_env = parent_env
    self._env_cache = Cache()  # Env cache so eval caching works more effectively

  def dict(self):
    return self.__items

  def get(self, key, default=None):
    if key in self:
      return self[key]
    return default

  def __getitem__(self, key):
    if type(key) == int:
      raise ValueError('Trying to access tuple as a list')

    x = self.get_thunk(key)

    # Check if this is a Thunk that needs to be lazily evaluated before we
    # return it.
    if isinstance(x, Thunk):
      return eval(x, self.env(self))

    return x

  def __contains__(self, key):
    return key in self.__items

  def env(self, current_scope):
    """Return an environment that will look up in current_scope for keys in
    this tuple, and the parent env otherwise.
    """
    return self._env_cache.get(
            current_scope.ident,
            lambda: Environment(current_scope, self._parent_env, names=self.keys()))

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
    if k not in self.__items:
      raise EvaluationError('Unknown key: %r' % k)
    x = self.__items[k]
    # Don't evaluate in this env but parent env
    if isinstance(x, Inherit):
      # Change an unbound Inherit into a bound Inherit
      return Inherit(k, self._parent_env)
    return x

  def _render(self, key):
    if key in self:
      return '%s = %r' % (key, self.get_thunk(key))
    else:
      return '%s' % key

  def compose(self, tup):
    if not isinstance(tup, Tuple):
      tup = Tuple(tup, EmptyEnvironment())
    return CompositeTuple(self.tuples + [tup])

  def __repr__(self):
    return '{%s}' % '; '.join(self._render(k) for k in self.keys())


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
        thunk = tup.get_thunk(key)
        if not isinstance(thunk, Thunk):
          return thunk
        if not isinstance(thunk, Void):
          return eval(thunk, env)
    raise EvaluationError('Unknown key in base: %r' % key)


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
  def __init__(self, tuples):
    self.ident = obj_ident()
    self._tuples = tuples
    self._keys = functools.reduce(lambda s, t: s.union(t.keys()), self._tuples, set())
    self._makeLookupList()

  def _makeLookupList(self):
    # Count index from the back because we're going to reverse
    envs = [Environment({'base': CompositeBaseTuple(self, len(self.tuples) - i)}, env_of(t, self)) for i, t in enumerate(self.tuples)]
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

  def compose(self, tup):
    if not isinstance(tup, Tuple):
      tup = Tuple(tup, EmptyEnvironment())
    return CompositeTuple(self.tuples + [tup])

  def __getitem__(self, key):
    for tup, env in self.lookups:
      if key in tup:
        thunk = tup.get_thunk(key)
        if not isinstance(thunk, Thunk):
          return thunk  # Not a thunk but a literal then
        if not isinstance(thunk, Void):
          return eval(thunk, env)
    raise EvaluationError('Unknown key: %r' % key)

  def __repr__(self):
    return ' '.join(repr(t) for t in self.tuples)


class Application(Thunk):
  """Function application."""
  def __init__(self, left, right):
    self.ident = obj_ident()
    self.left = left
    self.right = right

  def eval(self, env):
    fn = eval(self.left, env)
    arg = eval(self.right, env)

    # Normalize arg into a list of arguments, which it already is if the
    # right-hand side is an ArgList, but not otherwise.
    if not isinstance(self.right, ArgList):
      arg = [arg]

    # We now have evaluated and unevaluated versions of functor and arguments
    # The evaluated ones will be used for processing, the unevaluated ones will
    # be used for error reporting.

    # Tuple application
    if isinstance(fn, Tuple):
      return self.applyTuple(fn, arg, env)

    # List application
    if isinstance(fn, list):
      return self.applyList(fn, arg)

    # Any other callable type, just use as a Python function
    if not callable(fn):
      raise EvaluationError('Result of %r (%r) not callable' % (self.left, fn))

    if isinstance(fn, functions.EnvironmentFunction):
      return fn(*arg, env=env)

    return fn(*arg)

  def __repr__(self):
    return '%r(%r)' % (self.left, self.right)

  def applyTuple(self, tuple, right, env):
    """Apply a tuple to something else."""
    if len(right) != 1:
      raise EvaluationError('Tuple (%r) can only be applied to one argument, got %r' % (self.left, self.right))
    right = right[0]

    if isinstance(right, Tuple):
      return CompositeTuple(tuple.tuples + right.tuples)

    if is_str(right):
      return tuple[right]

    raise EvaluationError("Can't apply tuple (%r) to argument (%r): string or tuple expected, got %r" % (self.left, self.right, right))

  def applyList(self, lst, right):
    """Apply a list to something else."""
    if len(right) != 1:
      raise EvaluationError('List (%r) can only be applied to one argument, got %r' % (self.left, self.right))
    right = right[0]

    if isinstance(right, int):
      return lst[right]

    raise EvaluationError("Can't apply list (%r) to argument (%r): integer expected, got %r" % (self.left, self.right, right))


def mkApplications(atoms):
  """Make a sequence of applications from a list of tokens.

  atoms is a list of atoms, which will be handled left-associatively. E.g:

      ['foo', [], []] == foo()() ==> Application(Application('foo', []), [])
  """
  atoms = list(atoms)
  while len(atoms) > 1:
    atoms[0:2] = [Application(atoms[0], atoms[1])]

  # Nothing left to apply
  return atoms[0]


class UnOp(Thunk):
  def __init__(self, op, right):
    self.ident = obj_ident()
    self.op = op
    self.right = right

  def eval(self, env):
    right = eval(self.right, env)
    fn = functions.unary_operators.get(self.op, None)
    if fn is None:
      raise EvaluationError('Unknown unary operator: %s' % self.op)
    return fn(right)

  def __repr__(self):
    return '%s%r' % (self.op, self.right)


def mkUnOp(tokens):
  return UnOp(tokens[0], tokens[1])


class BinOp(Thunk):
  def __init__(self, left, op, right):
    self.ident = obj_ident()
    self.left = left
    self.op = op
    self.right = right

  def eval(self, env):
    left = eval(self.left, env)
    right = eval(self.right, env)

    fn = functions.all_binary_operators.get(self.op, None)
    if fn is None:
      raise EvaluationError('Unknown operator: %s' % self.op)

    return fn(left, right)

  def __repr__(self):
    return ('%r %s %r' % (self.left, self.op, self.right))


def mkBinOps(tokens):
  tokens = list(tokens)
  while len(tokens) > 1:
    assert(len(tokens) >= 3)
    tokens[0:3] = [BinOp(tokens[0], tokens[1], tokens[2])]
  return tokens[0]


class Deref(Thunk):
  """Dereferencing of a dictionary-like object."""
  def __init__(self, haystack, needle, location):
    self.ident = obj_ident()
    self.haystack = haystack
    self.needle = needle
    self.location = location

  def eval(self, env):
    try:
      haystack = eval(self.haystack, env)
      return haystack[self.needle]
    except EvaluationError as e:
      raise EvaluationError(self.location.error_in_context('while evaluating \'%r\'' % self), e)
    except TypeError as e:
      raise EvaluationError(self.location.error_in_context('while getting %r from %r' % (self.needle, self.haystack)), e)

  def __repr__(self):
    return '%s.%s' % (self.haystack, self.needle)


def mkDerefs(s, loc, tokens):
  location = SourceLocation(s, loc)
  tokens = list(tokens)
  while len(tokens) > 1:
    tokens[0:2] = [Deref(tokens[0], tokens[1], location)]
  return tokens[0]


class Condition(Thunk):
  def __init__(self, cond, then, else_):
    self.ident = obj_ident()
    self.cond = cond
    self.then = then
    self.else_ = else_

  def eval(self, env):
    if eval(self.cond, env):
      return eval(self.then, env)
    else:
      return eval(self.else_, env)

  def __repr__(self):
    return 'if %r then %r else %r' % (self.cond, self.then, self.else_)


class Include(Thunk):
  def __init__(self, file_ref):
    self.ident = obj_ident()
    self.file_ref = file_ref
    self.current_file = the_context.filename
    self.loader = the_context.loader

  def eval(self, env):
    file_ref = eval(self.file_ref, env)
    if not is_str(file_ref):
      raise EvaluationError('Included argument (%r) must be a string, got %r' %
                            (self.file_ref, file_ref))

    return self.loader(self.current_file, file_ref)

  def __repr__(self):
    return 'include(%r)' % self.file_ref


#----------------------------------------------------------------------
#  Grammar
#

def sym(sym):
  return p.Literal(sym).suppress()


def kw(kw):
  return p.Keyword(kw).suppress()


def listMembers(sep, expr, what):
  return p.Optional(p.delimitedList(expr, sep) -
                    p.Optional(sep).suppress()).setParseAction(
                        lambda ts: what(list(ts)))


def bracketedList(l, r, sep, expr, what):
  """Parse bracketed list.

  Empty list is possible, as is a trailing separator.
  """
  return (sym(l) - listMembers(sep, expr, what) - sym(r)).setParseAction(head)


keywords = ['and', 'or', 'not', 'if', 'then', 'else', 'include', 'inherit', 'null', 'true', 'false']

expression = p.Forward()

comment = '#' + p.restOfLine

identifier = p.Regex(r'[a-zA-Z_][a-zA-Z0-9_:-]*')

# Contants
integer = p.Word(p.nums).setParseAction(do(head, int, Constant))
floating = p.Regex(r'\d*\.\d+').setParseAction(do(head, float, Constant))
dq_string = p.QuotedString('"', escChar='\\', multiline=True).setParseAction(do(head, Constant))
sq_string = p.QuotedString("'", escChar='\\', multiline=True).setParseAction(do(head, Constant))
boolean = (p.Keyword('true') | p.Keyword('false')).setParseAction(do(head, mkBool, Constant))
null = p.Keyword('null').setParseAction(Null)

# List
list_ = bracketedList('[', ']', ',', expression, List)

# Tuple
inherit = (kw('inherit') - p.ZeroOrMore(identifier)).setParseAction(mkInherits)
tuple_member = (inherit
               | (identifier + ~p.FollowedBy('=')).setParseAction(lambda s, loc, x: (x[0], Void(x[0], SourceLocation(s, loc))))
               | (identifier - '=' - expression).setParseAction(lambda x: (x[0], x[2]))
               )
tuple_members = listMembers(';', tuple_member, UnboundTuple)
tuple = bracketedList('{', '}', ';', tuple_member, UnboundTuple)

# Variable (can't be any of the keywords, which may have lower matching priority)
variable = ~p.oneOf(' '.join(keywords)) + identifier.copy().setParseAction(mkVar)

# Argument list will live by itself as a atom. Actually, it's a tuple, but we
# don't call it that because we use that term for something else already :)
arg_list = bracketedList('(', ')', ',', expression, ArgList)

parenthesized_expr = (sym('(') - expression - ')').setParseAction(head)

unary_op = (p.oneOf(' '.join(functions.unary_operators.keys())) - expression).setParseAction(mkUnOp)

if_then_else = (kw('if') - expression -
                kw('then') - expression -
                kw('else') - expression).setParseAction(doapply(Condition))

# We don't allow space-application here
# Now our grammar is becoming very dirty and hackish
deref = p.Forward()
include = (kw('include') - deref).setParseAction(doapply(Include))

atom = (tuple
        | variable
        | dq_string
        | sq_string
        | boolean
        | list_
        | null
        | unary_op
        | parenthesized_expr
        | if_then_else
        | include
        | floating
        | integer
        )

# We have two different forms of function application, so they can have 2
# different precedences. This one: fn(args), which binds stronger than
# dereferencing (fn(args).attr == (fn(args)).attr)
applic1 = (atom - p.ZeroOrMore(arg_list)).setParseAction(mkApplications)

# Dereferencing of an expression (obj.bar)
deref << (applic1 - p.ZeroOrMore(p.Literal('.').suppress() - identifier)).setParseAction(mkDerefs)

# Juxtaposition function application (fn arg), must be 1-arg every time
applic2 = (deref - p.ZeroOrMore(deref)).setParseAction(mkApplications)

# All binary operators at various precedence levels go here:
# This piece of code does the moral equivalent of:
#
#     T = F*F | F/F | F
#     E = T+T | T-T | T
#
# etc.
term = applic2
for op_level in functions.binary_operators:
  operator_syms = ' '.join(op_level.keys())
  term = (term - p.ZeroOrMore(p.oneOf(operator_syms) - term)).setParseAction(mkBinOps)

expression << term

# Two entry points: start at an arbitrary expression, or expect the top-level
# scope to be a tuple.
start = expression.ignore(comment)
start_tuple = tuple_members.ignore(comment)

#----------------------------------------------------------------------
#  Top-level functions
#

default_env = Environment(functions.builtin_functions)

def reads(s, filename=None, loader=None, implicit_tuple=True):
  """Load but don't evaluate a GCL expression from a string."""
  try:
    the_context.filename = filename or '<input>'
    the_context.loader = loader or default_loader
    return (start_tuple if implicit_tuple else start).parseString(s, parseAll=True)[0]
  except (p.ParseException, p.ParseSyntaxException) as e:
    msg = '%s:%d: %s\n%s\n%s^-- here' % (the_context.filename, e.lineno, e.msg, e.line, ' ' * (e.col - 1))
    raise ParseError(msg)


def read(filename, loader=None, implicit_tuple=True):
  """Load but don't evaluate a GCL expression from a file."""
  with open(filename, 'r') as f:
    return reads(f.read(),
                 filename=filename,
                 loader=loader,
                 implicit_tuple=implicit_tuple)


def loads(s, filename=None, loader=None, implicit_tuple=True, env=None):
  """Load and evaluate a GCL expression from a string."""
  ast = reads(s, filename=filename, loader=loader, implicit_tuple=implicit_tuple)
  return eval(ast, env or default_env)


def load(filename, loader=None, implicit_tuple=True, env=None):
  """Load and evaluate a GCL expression from a file."""
  with open(filename, 'r') as f:
    return loads(f.read(),
                 filename=filename,
                 loader=loader,
                 implicit_tuple=implicit_tuple,
                 env=env)

