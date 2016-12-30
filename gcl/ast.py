"""
AST and parsing related functions.
"""
import collections
import itertools
import string
import sys
import textwrap

import pyparsing as p

from . import runtime
from . import exceptions
from . import schema
from . import functions
from . import framework


def do(*fns):
  def fg(args):
    for fn in fns:
      args = fn(args)
    return args
  return fg


def head(x): return x[0]
def second(x): return x[1]
def inner(x): return x[1:-1]
def mkBool(s): return True if s == 'true' else False
def drop(x): return []


def callParseAction(action, src_loc, tokens):
  try:
    return action(src_loc, *list(tokens))
  except TypeError as e:
    # pyparsing will catch TypeErrors to "detect" our arity, but I don't want to swallow errors
    # here. Convert to some other exception type. I'd LOVE to keep the stack trace here, but there
    # is no one syntax that is not a syntax error in either Python 2 or Python 3. So in Python 2
    # we don't keep the stack trace, unfortunately.

    if hasattr(e, 'with_traceback'):
      # Python 3
      raise RuntimeError(str(e)).with_traceback(sys.exc_info()[2])
    else:
      # Python 2, put the original trace inside the error message. This exception is never
      # supposed to happen anyway, it is just for debugging.
      import traceback
      raise RuntimeError(traceback.format_exc())


def parseWithLocation(expr, action):
  startMarker = p.Empty().setParseAction(lambda s, loc, t: loc)
  endMarker = startMarker.copy()
  complete = startMarker + expr + endMarker

  def parseAction(s, loc, t):
    start, inner_tokens, end = t[0], t[1:-1], t[-1]
    src_loc = SourceLocation(s, start, end)
    return callParseAction(action, src_loc, inner_tokens)

  complete.setParseAction(parseAction)
  return complete


def convertAndMake(converter, handler):
  """Convert with location."""
  def convertAction(loc, value):
    return handler(loc, converter(value))
  return convertAction


class ParseContext(object):
  def __init__(self):
    self.filename = '<from string>'
    self.loader = None

the_context = ParseContext()


class SourceLocation(object):
  def __init__(self, string, start_offset, end_offset=None):
    self.filename = the_context.filename
    self.string = string
    self.start_offset = start_offset
    self.end_offset = end_offset

  @property
  def line(self):
    return p.line(self.start_offset, self.string)

  @property
  def lineno(self):
    return p.lineno(self.start_offset, self.string)

  @property
  def end_lineno(self):
    assert self.end_offset is not None
    return p.lineno(self.end_offset, self.string)

  @property
  def col(self):
    return p.col(self.start_offset, self.string)

  @property
  def end_col(self):
    assert self.end_offset is not None
    return p.col(self.end_offset, self.string)

  @property
  def line_spec(self):
    return '%s:%s' % (self.filename, self.lineno)

  def error_in_context(self, msg):
    msg = '%s:%d: %s in \'%s\'' % (self.filename, self.lineno, msg, self.line)
    return msg

  def original_string(self):
    return self.string[self.start_offset:self.end_offset]

  def __str__(self):
    if self.end_offset:
      return self.string[:self.start_offset] + '[' + self.original_string() + ']' + self.string[self.end_offset:]
    else:
      return self.string[:self.start_offset] + '|' + self.string[self.start_offset:]

  @staticmethod
  def empty():
    return SourceLocation('', 0)

  def contains(self, q):
    return (q.filename == self.filename
        and (q.line > self.lineno or (q.line == self.lineno and q.col >= self.col))
        and (q.line < self.end_lineno or (q.line == self.end_lineno and q.col <= self.end_col)))

  def __repr__(self):
    return 'SourceLocation(%r, %r:%r, %r:%r)' % (self.filename, self.lineno, self.col, self.end_lineno, self.end_col)


class AstNode(object):
  def find_tokens(self, q):
    """Find all AST nodes at the given filename, line and column."""
    found_me = []
    if hasattr(self, 'location'):
      if self.location.contains(q):
        found_me = [self]
    elif self._found_by(q):
      found_me = [self]

    cs = [n.find_tokens(q) for n in self._children()]
    return found_me + list(itertools.chain(*cs))

  def _found_by(self, q):
    raise exceptions.EvaluationError('Not implemented')

  def _children(self):
    return []


class SourceQuery(object):
  def __init__(self, filename, line, col):
    self.filename = filename
    self.line = line
    self.col = col

  def __repr__(self):
    return 'SourceQuery(%r, %r, %r)' % (self.filename, self.line, self.col)


class Null(framework.Thunk, AstNode):
  """Null, evaluates to None."""
  def __init__(self, location, _):
    self.ident = framework.obj_ident()
    self.location = location

  def eval(self, env):
    return None

  def __repr__(self):
    return "null";


class Inherit(framework.BindableThunk, AstNode):
  """Inherit Thunks can be either bound or unbound."""

  def __init__(self, location, name, env=None):
    self.ident = framework.obj_ident()
    self.location = location
    self.name = name
    self.env = env

  def bind(self, env):
    return Inherit(self.location, self.name, env)

  def eval(self, env):
    if not self.env:
      raise exceptions.EvaluationError("Shouldn't evaluate unbound Inherit nodes")
    return self.env[self.name]

  def __repr__(self):
    return 'inherit %s' % self.name


def inheritNodes(tokens):
  for t in tokens:
    assert isinstance(t, Var)
  return [TupleMemberNode(t.location, t.name, no_schema, Inherit(t.location, t.name)) for t in list(tokens)]


class Literal(framework.Thunk, AstNode):
  """A GCL literal expression."""
  def __init__(self, location, value):
    self.ident = framework.obj_ident()
    self.value = value
    self.location = location

  def eval(self, env):
    return self.value

  def __repr__(self):
    if type(self.value) == bool:
      return 'true' if self.value else 'false'
    return repr(self.value)


class Var(framework.Thunk, AstNode):
  """Reference to another value."""
  def __init__(self, location, name):
    self.ident = framework.obj_ident()
    self.name = name
    self.location = location

  def eval(self, env):
    try:
      return env[self.name]
    except exceptions.EvaluationError as e:
      raise exceptions.EvaluationError(self.location.error_in_context('while evaluating %r' % self.name), e)

  def __repr__(self):
    return self.name


class List(framework.Thunk, AstNode):
  """A GCL list."""
  def __init__(self, location, *values):
    self.ident = framework.obj_ident()
    self.location = location
    self.values = list(values)

  def eval(self, env):
    return [framework.eval(v, env) for v in self.values]

  def __repr__(self):
    return repr(self.values)

  def _children(self):
    return self.values


class ArgList(framework.Thunk, AstNode):
  """A paren-separated argument list.

  This is actually a shallow wrapper for Python's list type. We can't use that
  because pyparsing will automatically concatenate lists, which we don't want
  in this case.
  """
  def __init__(self, values):
    self.ident = framework.obj_ident()
    self.values = values

  def eval(self, env):
    return [framework.eval(v, env) for v in self.values]

  def __repr__(self):
    return '(%s)' % ', '.join(repr(x) for x in self.values)

  def _children(self):
    return self.values

  def _found_by(self, q):
    return False


class TupleMemberNode(AstNode):
  """AST node for tuple members.

  They have a name, an expression value and an optional schema.
  """
  def __init__(self, location, name, schema, value=None):
    self.location = location
    self.name = name
    self.value = value
    self.member_schema = schema
    self.comment = DocComment(location)

    # Hack: set the name on the value if it is a Void value, to help the Void
    # generate better error messages. Not doing it here makes the grammar a lot
    # messier.
    if isinstance(self.value, Void):
      self.value.name = name


  def attach_comment(self, comment):
    self.comment = comment

  def __repr__(self):
    schema_repr = ': %r' % self.member_schema if not isinstance(self.member_schema, NoSchemaNode) else ''
    return '%s%s = %r' % (self.name, schema_repr, self.value)

  def _children(self):
    return [self.value]


class DocComment(object):
  def __init__(self, location, *lines):
    self.location = location
    self.lines = []
    self.tags = collections.defaultdict(lambda: '')

    lines = textwrap.dedent('\n'.join(lines)).split('\n')
    for line in lines:
      if line.startswith('@'):
        try:
          tag, content = line[1:].split(' ', 1)
        except ValueError:
          tag, content = line[1:], ''
        self.tags[tag] = content
      else:
        self.lines.append(line)

  def tag(self, name):
    return self.tags.get(name, None)

  def has_tag(self, name):
    return self.tag(name) is not None

  def title(self):
    return ' '.join(s.strip() for s in itertools.takewhile(nonempty, self.lines))

  def body_lines(self):
    return list(drop(1, itertools.dropwhile(nonempty, self.lines)))

  def __repr__(self):
    return '\n'.join('#. %s' % l for l in self.lines)


def nonempty(s):
  """Return True iff the given string is nonempty."""
  return len(s.strip())


def strip_space(s):
  return s[1:] if s.startswith(' ') else s


def drop(n, xs):
  for i, x in enumerate(xs):
    if n <= i:
      yield x


def attach_doc_comment(_, comment, member):
  member.attach_comment(comment)
  return member


def is_tuple_member(x):
  return isinstance(x, TupleMemberNode)


class TupleNode(framework.Thunk, AstNode):
  """AST node for tuple

  When evaluating, the tuple doesn't actually evaluate its children. Instead, we return a (lazy)
  Tuple object that only evaluates the elements when they're requested.
  """
  def __init__(self, location, *members):
    # Filter contents down to actual members (ignore UnparseableNodes)
    members = [m for m in members if is_tuple_member(m)]

    duplicates = [name for name, ns in itertools.groupby(sorted(m.name for m in members)) if len(list(ns)) > 1]
    if duplicates:
      raise exceptions.ParseError(the_context.filename, location, 'Key %s occurs more than once in tuple at %s' % (', '.join(duplicates), location.error_in_context('')))

    self.ident = framework.obj_ident()
    self.members = members
    self.member = {m.name: m for m in self.members}
    self.location = location
    self._cache = framework.Cache()

  def eval(self, env):
    return self._cache.get(env.ident, self._make_tuple, env)

  def _make_tuple(self, env):
    """Instantiate the Tuple based on this TupleNode."""
    t = runtime.Tuple(self, env, dict2tuple)
    # A tuple also provides its own schema spec
    schema = schema_spec_from_tuple(t)
    t.attach_schema(schema)
    return t

  def _children(self):
    return self.members

  def __repr__(self):
    return ('{' +
            '; '.join(repr(m) for m in self.members) +
            '}')


class RuntimeTupleNode(TupleNode):
  """Fake AST node.

  Tuples require a reference to a TupleNode-like object. However, sometimes we want to convert
  dictionaries to Tuple objects at runtime, so we need to invent a tuple-like object.
  """
  def __init__(self, dct):
    self.members = [TupleMemberNode(SourceLocation.empty(), key, schema=NoSchemaNode(), value=value) for key, value in dct.items()]
    self.member = {m.name: m for m in self.members}


def dict2tuple(dct):
    return runtime.Tuple(RuntimeTupleNode(dct), framework.EmptyEnvironment(), dict2tuple)


class Application(framework.Thunk, AstNode):
  """Function application."""
  def __init__(self, location, left, right):
    self.location = location
    self.ident = framework.obj_ident()
    self.left = left
    self.right = right

  def right_as_list(self):
    return self.right if isinstance(self.right, ArgList) else ArgList([self.right])

  def eval_right_as_list(self, env):
    return framework.eval(self.right_as_list(), env)

  def eval(self, env):
    fn = framework.eval(self.left, env)

    try:
      # Tuple application
      if isinstance(fn, framework.TupleLike):
        return self.applyTuple(fn, self.eval_right_as_list(env), env)

      # List application
      if isinstance(fn, list) or framework.is_str(fn):
        return self.applyIndex(fn, self.eval_right_as_list(env))

      # Any other callable type, just use as a Python function
      if not callable(fn):
        raise exceptions.EvaluationError('Result of %r (%r) not callable' % (self.left, fn))

      return call_fn(fn, self.right_as_list(), env)
    except Exception as e:
      # Wrap exceptions
      raise exceptions.EvaluationError(self.location.error_in_context('while calling \'%r\'' % self), e)

  def _children(self):
    return [self.left, self.right]

  def __repr__(self):
    return '%r(%r)' % (self.left, self.right)

  def applyTuple(self, tuple, right, env):
    """Apply a tuple to something else."""
    if len(right) != 1:
      raise exceptions.EvaluationError('Tuple (%r) can only be applied to one argument, got %r' % (self.left, self.right))
    right = right[0]

    return tuple(right)

  def applyIndex(self, lst, right):
    """Apply a list to something else."""
    if len(right) != 1:
      raise exceptions.EvaluationError('%r can only be applied to one argument, got %r' % (self.left, self.right))
    right = right[0]

    if isinstance(right, int):
      return lst[right]

    raise exceptions.EvaluationError("Can't apply %r to argument (%r): integer expected, got %r" % (self.left, self.right, right))


def mkApplications(location, *atoms):
  """Make a sequence of applications from a list of tokens.

  atoms is a list of atoms, which will be handled left-associatively. E.g:

      ['foo', [], []] == foo()() ==> Application(Application('foo', []), [])
  """
  atoms = list(atoms)
  while len(atoms) > 1:
    atoms[0:2] = [Application(location, atoms[0], atoms[1])]

  # Nothing left to apply
  return atoms[0]


class UnOp(framework.Thunk, AstNode):
  def __init__(self, op, right):
    self.ident = framework.obj_ident()
    self.op = op
    self.right = right

  def _found_by(self, q):
    return False

  def eval(self, env):
    fn = functions.unary_operators.get(self.op, None)
    if fn is None:
      raise exceptions.EvaluationError('Unknown unary operator: %s' % self.op)
    return call_fn(fn, ArgList([self.right]), env)

  def __repr__(self):
    return '%s%s%r' % (self.op, ' ' if self.op == 'not' else '', self.right)


def mkUnOp(tokens):
  return UnOp(tokens[0], tokens[1])


class BinOp(framework.Thunk):
  def __init__(self, left, op, right):
    self.ident = framework.obj_ident()
    self.left = left
    self.op = op
    self.right = right

  def _found_by(self, q):
    return False

  def eval(self, env):
    fn = functions.all_binary_operators.get(self.op, None)
    if fn is None:
      raise exceptions.EvaluationError('Unknown operator: %s' % self.op)

    return call_fn(fn, ArgList([self.left, self.right]), env)

  def __repr__(self):
    return ('%r %s %r' % (self.left, self.op, self.right))


def mkBinOps(tokens):
  tokens = list(tokens)
  while len(tokens) > 1:
    assert(len(tokens) >= 3)
    tokens[0:3] = [BinOp(tokens[0], tokens[1], tokens[2])]
  return tokens[0]


def call_fn(fn, arglist, env):
  """Call a function, respecting all the various types of functions that exist."""
  if isinstance(fn, framework.LazyFunction):
    # The following looks complicated, but this is necessary because you can't
    # construct closures over the loop variable directly.
    thunks = [(lambda thunk: lambda: framework.eval(thunk, env))(th) for th in arglist.values]
    return fn(*thunks)

  evaled_args = framework.eval(arglist, env)
  if isinstance(fn, framework.EnvironmentFunction):
    return fn(*evaled_args, env=env)

  return fn(*evaled_args)


class Deref(framework.Thunk, AstNode):
  """Dereferencing of a dictionary-like object."""
  def __init__(self, location, haystack, needle):
    self.location = location
    self.ident = framework.obj_ident()
    self.haystack = haystack
    self.needle = needle

  def _children(self):
    return [self.haystack, self.needle]

  def eval(self, env):
    try:
      haystack = framework.eval(self.haystack, env)
      return haystack[self.needle]
    except exceptions.EvaluationError as e:
      raise exceptions.EvaluationError(self.location.error_in_context('while evaluating \'%r\'' % self), e)
    except TypeError as e:
      raise exceptions.EvaluationError(self.location.error_in_context('while getting %r from %r' % (self.needle, self.haystack)), e)

  def __repr__(self):
    return '%s.%s' % (self.haystack, self.needle)


def mkDerefs(location, *tokens):
  tokens = list(tokens)
  while len(tokens) > 1:
    tokens[0:2] = [Deref(location, tokens[0], tokens[1])]
  return tokens[0]


class Condition(framework.Thunk, AstNode):
  def __init__(self, _, cond, then, else_):
    self.ident = framework.obj_ident()
    self.cond = cond
    self.then = then
    self.else_ = else_

  def _children(self):
    return [self.cond, self.then, self.else_]

  def _found_by(self, q):
    return False

  def eval(self, env):
    if framework.eval(self.cond, env):
      return framework.eval(self.then, env)
    else:
      return framework.eval(self.else_, env)

  def __repr__(self):
    return 'if %r then %r else %r' % (self.cond, self.then, self.else_)


class ListComprehension(framework.Thunk, AstNode):
  def __init__(self, location, expr, var, collection, cond=None):
    self.ident = framework.obj_ident()
    self.expr = expr
    self.var = var
    self.collection = collection
    self.cond = cond

  def _children(self):
    return [self.expr, self.var, self.collection, self.cond]

  def eval(self, env):
    ret = []
    for x in framework.eval(self.collection, env):
      new_env = framework.Environment({self.var.name:x}, env)
      if self.cond is None or framework.eval(self.cond, new_env):
        ret.append(framework.eval(self.expr, new_env))
    return ret

  def __repr__(self):
    return '[%r for %r in %r]' % (self.expr, self.var, self.collection)


class UnparseableNode(framework.Thunk, AstNode):
  """An unparseable exception."""
  def __init__(self, location):
    self.location = location
    self.ident = framework.obj_ident()

  def eval(self, env):
    raise exceptions.EvaluationError(self.location.error_in_context('Unparseable expression'))

  def __repr__(self):
    return self.location.original_string()


class Void(framework.Thunk, AstNode):
  """A missing value."""
  def __init__(self, location, name):
    self.location = location
    self.name = name
    self.ident = framework.obj_ident()

  def _found_by(self, q):
    return False

  def eval(self, env):
    raise exceptions.EvaluationError(self.location.error_in_context('Unbound value: %r' % self.name))

  def is_unbound(self):
    return True

  def __repr__(self):
    return '<unbound>'


class Include(framework.Thunk, AstNode):
  def __init__(self, location, file_ref):
    self.ident = framework.obj_ident()
    self.file_ref = file_ref
    self.current_file = the_context.filename
    self.loader = the_context.loader

  def _children(self):
    return [self.file_ref]

  def eval(self, env):
    file_ref = framework.eval(self.file_ref, env)
    if not framework.is_str(file_ref):
      raise exceptions.EvaluationError('Included argument (%r) must be a string, got %r' %
                            (self.file_ref, file_ref))

    return self.loader(self.current_file, file_ref, env=env.root)

  def __repr__(self):
    return 'include(%r)' % self.file_ref

#----------------------------------------------------------------------
#  Schema AST model
#

class NoSchemaNode(framework.Thunk):
  """For values without a schema."""
  def __init__(self):
    self.ident = framework.obj_ident()
    self.required = False
    self.private = False

  def eval(self, env):
    return schema.AnySchema()

no_schema = NoSchemaNode()  # Singleton object


class AnySchemaExprNode(framework.Thunk):
  def __init__(self):
    self.ident = framework.obj_ident()

  def eval(self, env):
    return schema.AnySchema()

any_schema_expr = AnySchemaExprNode()


class MemberSchemaNode(framework.Thunk):
  """AST node for member schema definitions. Can be evaluated to produce runtime Schema classes.

  NOTE: this class does a little funky logic. Because we want to be able to write:

      var : int;

  Instead of 'var : "int"', some words are special-cased. However, we also want to be able to parse
  things of the form:

      tuple_with_int_foo : { foo : int };
      tuple_of_type : SomeObject;

  Which requires us to parse the grammar 'expression+schema_literals' (as opposed to just the
  grammar 'expression'), which I don't really want to double-define. Instead, we'll just pretend
  certain variable names like 'int' and 'string' refer to global objects that can't be rebound (in
  effect, they resolve to their variable name).

  The schema evaluation object model looks like this:


      TupleNode <>------> TupleMemberNode <>---------> MemberSchemaNode
         |                                                   |
         | eval()                                            | eval()
         v                                                   v
      Tuple                                           schema.Schema() +validate()
                                                             /\
                                                        TupleSchema


  """
  def __init__(self, location, private, required, expr):
    self.location = location
    self.private = private
    self.required = required
    self.expr = expr
    self.ident = framework.obj_ident()

  def eval(self, env):
    return make_schema_from(self.expr, env)

  def __repr__(self):
    return ' '.join((['required'] if self.required else []) +
                    ([repr(self.expr)] if not isinstance(self.expr, NoSchemaNode) else []))


class TupleSchemaAccess(object):
  """A class that behaves like a dictionary and returns member schemas from a tuple."""
  def __init__(self, tuple):
    self.tuple = tuple

  def __getitem__(self, key):
    return self.tuple.get_schema_spec(key)

  def __contains__(self, key):
    return key in self.tuple

  def get(self, key, default=None):
    return self[key] if key in self else default

  def values(self):
    return [self[k] for k in self.tuple.keys()]

  def __repr__(self):
    return 'TupleSchemaAccess(%r)' % self.tuple


def schema_spec_from_tuple(tup):
  """Return the schema spec from a run-time tuple."""
  if hasattr(tup, 'get_schema_spec'):
    # Tuples have a TupleSchema field that contains a model of the schema
    return schema.from_spec({
        'fields': TupleSchemaAccess(tup),
        'required': tup.get_required_fields()})
  return schema.AnySchema()


def make_schema_from(value, env):
  """Make a Schema object from the given spec.

  The input and output types of this function are super unclear, and are held together by ponies,
  wishes, duct tape, and a load of tests. See the comments for horrific entertainment.
  """

  # So this thing may not need to evaluate anything[0]
  if isinstance(value, framework.Thunk):
    value = framework.eval(value, env)

  # We're a bit messy. In general, this has evaluated to a Schema object, but not necessarily:
  # for tuples and lists, we still need to treat the objects as specs.
  if isinstance(value, schema.Schema):
    return value

  if framework.is_tuple(value):
    # If it so happens that the thing is a tuple, we need to pass in the data in a bit of a
    # different way into the schema factory (in a dictionary with {fields, required} keys).
    return schema_spec_from_tuple(value)

  if framework.is_list(value):
    # [0] This list may contain tuples, which oughta be treated as specs, or already-resolved schema
    # objects (as returned by 'int' and 'string' literals). make_schema_from
    # deals with both.
    return schema.from_spec([make_schema_from(x, env) for x in value])

  raise exceptions.EvaluationError('Can\'t make a schema from %r' % value)


def make_tuple(x):
  """Turn a dict-like object into a Tuple."""
  if isinstance(x, framework.TupleLike):
    return x
  return dict2tuple(x)

#----------------------------------------------------------------------
#  Grammar
#

def sym(sym):
  return p.Suppress(sym)


def kw(kw):
  return p.Keyword(kw).suppress()


def listMembers(sep, expr):
  return p.Optional(p.delimitedList(expr, sep) -
                    p.Optional(sep).suppress())


def bracketedList(l, r, sep, expr, allow_missing_close=False):
  """Parse bracketed list.

  Empty list is possible, as is a trailing separator.
  """
  closer = sym(r) if not allow_missing_close else p.Optional(sym(r))
  return (sym(l) - listMembers(sep, expr) - closer)


def unquote(s):
  """Unquote the indicated string."""
  # Ignore the left- and rightmost chars (which should be quotes).
  # Use the Python engine to decode the escape sequence
  i, N = 1, len(s) - 1
  ret = []
  while i < N:
    if s[i] == '\\' and i < N - 1:
      ret.append(UNQUOTE_MAP.get(s[i+1], s[i+1]))
      i += 2
    else:
      ret.append(s[i])
      i += 1
  return ''.join(ret)


UNQUOTE_MAP = {
    'n' : '\n',
    'r' : '\r',
    't' : '\t',
    }


GRAMMAR_CACHE = {}
def make_grammar(allow_errors):
  """Make the part of the grammar that depends on whether we swallow errors or not."""
  if allow_errors in GRAMMAR_CACHE:
    return GRAMMAR_CACHE[allow_errors]

  def swallow_errors(rule, synchronizing_tokens):
    """Extend the production rule by potentially eating errors.

    This does not return a p.NoMatch() because that messes up the error messages.
    """
    if allow_errors:
      rule = rule | parseWithLocation(p.Suppress(p.CharsNotIn(synchronizing_tokens, min=1)), UnparseableNode)
    return rule

  class Grammar:
    keywords = ['and', 'or', 'not', 'if', 'then', 'else', 'include', 'inherit', 'null', 'true', 'false',
        'for', 'in']

    expression = p.Forward()

    comment =  p.Regex('#') + ~p.FollowedBy(sym('.')) + p.restOfLine
    doc_comment = (sym('#.') + p.restOfLine)

    quotedIdentifier = p.QuotedString('`', multiline=False)

    # - Must start with an alphascore
    # - May contain alphanumericscores and special characters such as : and -
    # - Must not end in a special character
    identifier = quotedIdentifier | p.Regex(r'[a-zA-Z_]([a-zA-Z0-9_:-]*[a-zA-Z0-9_])?')

    # Variable identifier (can't be any of the keywords, which may have lower matching priority)
    variable = ~p.Or([p.Keyword(k) for k in keywords]) + parseWithLocation(identifier.copy(), Var)

    # Contants
    integer = parseWithLocation(p.Word(p.nums), convertAndMake(int, Literal))
    floating = parseWithLocation(p.Regex(r'\d*\.\d+'), convertAndMake(float, Literal))
    dq_string = parseWithLocation(p.QuotedString('"', escChar='\\', unquoteResults=False, multiline=True), convertAndMake(unquote, Literal))
    sq_string = parseWithLocation(p.QuotedString("'", escChar='\\', unquoteResults=False, multiline=True), convertAndMake(unquote, Literal))
    boolean = parseWithLocation(p.Keyword('true') | p.Keyword('false'), convertAndMake(mkBool, Literal))
    null = parseWithLocation(p.Keyword('null'), Null)

    # List
    list_ = parseWithLocation(bracketedList('[', ']', ',', expression), List)

    # Tuple
    inherit = (kw('inherit') - p.ZeroOrMore(variable)).setParseAction(inheritNodes)
    schema_spec = parseWithLocation(p.Optional(p.Keyword('private').setParseAction(lambda: True), default=False)
                  - p.Optional(p.Keyword('required').setParseAction(lambda: True), default=False)
                  - p.Optional(swallow_errors(expression, ';}'), default=any_schema_expr), MemberSchemaNode)
    optional_schema = p.Optional(p.Suppress(':') - swallow_errors(schema_spec, '=;}'), default=no_schema)

    expression_value = sym('=') - swallow_errors(expression, ';}')
    void_value = parseWithLocation(p.FollowedBy(sym(';') | sym('}')), lambda loc: Void(loc, 'nonameyet'))
    member_value = expression_value | void_value
    named_member = parseWithLocation(identifier - optional_schema - member_value, TupleMemberNode)
    documented_member = parseWithLocation(parseWithLocation(p.ZeroOrMore(doc_comment), DocComment) + named_member, attach_doc_comment)
    tuple_member = swallow_errors(inherit | documented_member, ';}')

    tuple_members = parseWithLocation(listMembers(';', tuple_member), TupleNode)
    tuple = parseWithLocation(bracketedList('{', '}', ';', tuple_member, allow_missing_close=allow_errors), TupleNode)

    # Argument list will live by itself as a atom. Actually, it's a tuple, but we
    # don't call it that because we use that term for something else already :)
    arg_list = bracketedList('(', ')', ',', expression).setParseAction(ArgList)

    parenthesized_expr = (sym('(') - expression - ')').setParseAction(head)

    unary_op = (p.oneOf(' '.join(functions.unary_operators.keys())) - expression).setParseAction(mkUnOp)

    if_then_else = parseWithLocation(kw('if') + expression +
                    kw('then') - expression -
                    kw('else') - expression, Condition)

    list_comprehension = parseWithLocation(sym('[') + expression + kw('for') - variable - kw('in') -
        expression - p.Optional(kw('if') - expression) - sym(']'), ListComprehension)


    # We don't allow space-application here
    # Now our grammar is becoming very dirty and hackish
    deref = p.Forward()
    include = parseWithLocation(kw('include') - deref, Include)

    atom = (tuple
            | variable
            | dq_string
            | sq_string
            | boolean
            | list_comprehension
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
    applic1 = parseWithLocation(atom - p.ZeroOrMore(arg_list), mkApplications)

    # Dereferencing of an expression (obj.bar)
    deref << parseWithLocation(applic1 - p.ZeroOrMore(p.Suppress('.') - identifier), mkDerefs)

    # All binary operators at various precedence levels go here:
    # This piece of code does the moral equivalent of:
    #
    #     T = F*F | F/F | F
    #     E = T+T | T-T | T
    #
    # etc.
    term = deref
    for op_level in functions.binary_operators_before_juxtaposition:
      operator_syms = ' '.join(op_level.keys())
      term = (term - p.ZeroOrMore(p.oneOf(operator_syms) - term)).setParseAction(mkBinOps)

    # Juxtaposition function application (fn arg), must be 1-arg every time
    applic2 = parseWithLocation(term - p.ZeroOrMore(term), mkApplications)

    term = applic2
    for op_level in functions.binary_operators_after_juxtaposition:
      operator_syms = ' '.join(op_level.keys())
      term = (term - p.ZeroOrMore(p.oneOf(operator_syms) - term)).setParseAction(mkBinOps)

    expression << term

    # Two entry points: start at an arbitrary expression, or expect the top-level
    # scope to be a tuple.
    start = expression.ignore(comment)
    start_tuple = tuple_members.ignore(comment)
  GRAMMAR_CACHE[allow_errors] = Grammar
  return Grammar


def normal_grammar(): return make_grammar(False)
def lenient_grammar(): return make_grammar(True)


def find_offset(s, line, col):
  c_line = 1
  c_col = 1
  for i in range(len(s)):
    if (c_line == line and c_col >= col) or c_line > line:
      return i
    if s[i] == '\n':
      c_col = 1
      c_line += 1
    else:
      c_col += 1
  return len(s)


def reads(s, filename, loader, implicit_tuple, allow_errors):
  """Load but don't evaluate a GCL expression from a string."""
  try:
    the_context.filename = filename
    the_context.loader = loader

    grammar = make_grammar(allow_errors=allow_errors)
    root = grammar.start_tuple if implicit_tuple else grammar.start

    return root.parseWithTabs().parseString(s, parseAll=True)[0]
  except (p.ParseException, p.ParseSyntaxException) as e:
    loc = SourceLocation(s, find_offset(s, e.lineno, e.col))
    raise exceptions.ParseError(the_context.filename, loc, e.msg)
