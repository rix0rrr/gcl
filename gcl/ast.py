"""
AST and parsing related functions.
"""
import itertools
import sys

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


def doapply(what):
  def fn(args):
    return what(*args)
  return fn


def head(x): return x[0]
def second(x): return x[1]
def inner(x): return x[1:-1]
def mkBool(s): return True if s == 'true' else False
def drop(x): return []


def pafac(fn):
  """Make a function that accepts a parsed string and a SourceLocation into a function that can be
  passed into a setParseAction.
  """
  def wrapped(s, loc, x):
    try:
      return fn(SourceLocation(s, loc), *list(x))
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
        raise RuntimeError(traceback.format_exc())
  return wrapped


class ParseContext(object):
  def __init__(self):
    self.filename = '<from string>'
    self.loader = None

the_context = ParseContext()


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

  def __str__(self):
    return self.string[:self.offset] + '|' + self.string[self.offset:]

  @staticmethod
  def empty():
    return SourceLocation('', 0)


class Null(framework.Thunk):
  """Null, evaluates to None."""
  def __init__(self):
    self.ident = framework.obj_ident()

  def eval(self, env):
    return None

  def __repr__(self):
    return "null";


class Inherit(framework.BindableThunk):
  """Inherit Thunks can be either bound or unbound."""

  def __init__(self, name, env=None):
    self.ident = framework.obj_ident()
    self.name = name
    self.env = env

  def bind(self, env):
    return Inherit(self.name, env)

  def eval(self, env):
    if not self.env:
      raise exceptions.EvaluationError("Shouldn't evaluate unbound Inherit nodes")
    return self.env[self.name]

  def __repr__(self):
    return 'inherit %s' % self.name


def mkInherits(tokens):
  return [TupleMemberNode(SourceLocation('', 0), t, no_schema, Inherit(t)) for t in list(tokens)]


class Literal(framework.Thunk):
  """A GCL literal expression."""
  def __init__(self, value):
    self.ident = framework.obj_ident()
    self.value = value

  def eval(self, env):
    return self.value

  def __repr__(self):
    if type(self.value) == bool:
      return 'true' if self.value else 'false'
    return repr(self.value)


class Var(framework.Thunk):
  """Reference to another value."""
  def __init__(self, name, location):
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


def mkVar(s, loc, toks):
  return Var(toks[0], SourceLocation(s, loc))


class List(framework.Thunk):
  """A GCL list."""
  def __init__(self, values):
    self.ident = framework.obj_ident()
    self.values = list(values)

  def eval(self, env):
    return [framework.eval(v, env) for v in self.values]

  def __repr__(self):
    return repr(self.values)


class ArgList(framework.Thunk):
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


class TupleMemberNode(object):
  """AST node for tuple members.

  They have a name, an expression value and an optional schema.
  """
  def __init__(self, sloc, name, schema, value=None):
    self.sloc = sloc
    self.name = name
    self.value = value
    self.member_schema = schema

  def __repr__(self):
    schema_repr = ': %r' % self.member_schema if not isinstance(self.member_schema, NoSchemaNode) else ''
    return '%s%s = %r' % (self.name, schema_repr, self.value)


class TupleNode(framework.Thunk):
  """AST node for tuple

  When evaluating, the tuple doesn't actually evaluate its children. Instead, we return a (lazy)
  Tuple object that only evaluates the elements when they're requested.
  """
  def __init__(self, sloc, *members):
    duplicates = [name for name, ns in itertools.groupby(sorted(m.name for m in members)) if len(list(ns)) > 1]
    if duplicates:
      raise exceptions.ParseError('Key %s occurs more than once in tuple at %s' % (', '.join(duplicates), sloc.error_in_context('')))

    self.ident = framework.obj_ident()
    self.members = members
    self.member = {m.name: m for m in self.members}
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


class Application(framework.Thunk):
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


def mkApplications(s, loc, atoms):
  """Make a sequence of applications from a list of tokens.

  atoms is a list of atoms, which will be handled left-associatively. E.g:

      ['foo', [], []] == foo()() ==> Application(Application('foo', []), [])
  """
  location = SourceLocation(s, loc)
  atoms = list(atoms)
  while len(atoms) > 1:
    atoms[0:2] = [Application(location, atoms[0], atoms[1])]

  # Nothing left to apply
  return atoms[0]


class UnOp(framework.Thunk):
  def __init__(self, op, right):
    self.ident = framework.obj_ident()
    self.op = op
    self.right = right

  def eval(self, env):
    fn = functions.unary_operators.get(self.op, None)
    if fn is None:
      raise exceptions.EvaluationError('Unknown unary operator: %s' % self.op)
    return call_fn(fn, ArgList([self.right]), env)

  def __repr__(self):
    return '%s%r' % (self.op, self.right)


def mkUnOp(tokens):
  return UnOp(tokens[0], tokens[1])


class BinOp(framework.Thunk):
  def __init__(self, left, op, right):
    self.ident = framework.obj_ident()
    self.left = left
    self.op = op
    self.right = right

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


class Deref(framework.Thunk):
  """Dereferencing of a dictionary-like object."""
  def __init__(self, haystack, needle, location):
    self.ident = framework.obj_ident()
    self.haystack = haystack
    self.needle = needle
    self.location = location

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


def mkDerefs(s, loc, tokens):
  location = SourceLocation(s, loc)
  tokens = list(tokens)
  while len(tokens) > 1:
    tokens[0:2] = [Deref(tokens[0], tokens[1], location)]
  return tokens[0]


class Condition(framework.Thunk):
  def __init__(self, cond, then, else_):
    self.ident = framework.obj_ident()
    self.cond = cond
    self.then = then
    self.else_ = else_

  def eval(self, env):
    if framework.eval(self.cond, env):
      return framework.eval(self.then, env)
    else:
      return framework.eval(self.else_, env)

  def __repr__(self):
    return 'if %r then %r else %r' % (self.cond, self.then, self.else_)


class ListComprehension(framework.Thunk):
  def __init__(self, expr, var, collection, cond=None):
    self.ident = framework.obj_ident()
    self.expr = expr
    self.var = var
    self.collection = collection
    self.cond = cond

  def eval(self, env):
    ret = []
    for x in framework.eval(self.collection, env):
      new_env = framework.Environment({self.var.name:x}, env)
      if self.cond is None or framework.eval(self.cond, new_env):
        ret.append(framework.eval(self.expr, new_env))
    return ret

  def __repr__(self):
    return '[%r for %r in %r]' % (self.expr, self.var, self.collection)


class Void(framework.Thunk):
  """A missing value."""
  def __init__(self, name, location):
    self.name = name
    self.location = location
    self.ident = framework.obj_ident()

  def eval(self, env):
    raise exceptions.EvaluationError(self.location.error_in_context('Unbound value: %r' % self.name))

  def is_unbound(self):
    return True

  def __repr__(self):
    return '<unbound>'


class Include(framework.Thunk):
  def __init__(self, file_ref):
    self.ident = framework.obj_ident()
    self.file_ref = file_ref
    self.current_file = the_context.filename
    self.loader = the_context.loader

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

any_scheam_expr = AnySchemaExprNode()


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
  def __init__(self, sloc, private, required, expr):
    self.sloc = sloc
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


def listMembers(sep, expr, what):
  return p.Optional(p.delimitedList(expr, sep) -
                    p.Optional(sep).suppress()).setParseAction(what)


def bracketedList(l, r, sep, expr, what):
  """Parse bracketed list.

  Empty list is possible, as is a trailing separator.
  """
  return (sym(l) - listMembers(sep, expr, what) - sym(r)).setParseAction(head)


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
    't' : '\t',
    }


keywords = ['and', 'or', 'not', 'if', 'then', 'else', 'include', 'inherit', 'null', 'true', 'false',
    'for', 'in']

expression = p.Forward()

comment = '#' + p.restOfLine

quotedIdentifier = p.QuotedString('`', multiline=False)

# - Must start with an alphascore
# - May contain alphanumericscores and special characters such as : and -
# - Must not end in a special character
identifier = quotedIdentifier | p.Regex(r'[a-zA-Z_]([a-zA-Z0-9_:-]*[a-zA-Z0-9_])?')

# Contants
integer = p.Word(p.nums).setParseAction(do(head, int, Literal))
floating = p.Regex(r'\d*\.\d+').setParseAction(do(head, float, Literal))
dq_string = p.QuotedString('"', escChar='\\', unquoteResults=False, multiline=True).setParseAction(do(head, unquote, Literal))
sq_string = p.QuotedString("'", escChar='\\', unquoteResults=False, multiline=True).setParseAction(do(head, unquote, Literal))
boolean = (p.Keyword('true') | p.Keyword('false')).setParseAction(do(head, mkBool, Literal))
null = p.Keyword('null').setParseAction(Null)

# List
list_ = bracketedList('[', ']', ',', expression, List)

# Tuple
inherit = (kw('inherit') - p.ZeroOrMore(identifier)).setParseAction(mkInherits)
schema_spec = (p.Optional(p.Keyword('private').setParseAction(lambda: True), default=False)
               - p.Optional(p.Keyword('required').setParseAction(lambda: True), default=False)
               - p.Optional(expression, default=any_scheam_expr)).setParseAction(pafac(MemberSchemaNode))
optional_schema = p.Optional(p.Suppress(':') - schema_spec, default=no_schema)
tuple_member = (inherit
               | (identifier + optional_schema + ~p.FollowedBy('=')).setParseAction(lambda s, loc, x: TupleMemberNode(SourceLocation(s, loc), x[0], x[1], Void(x[0], SourceLocation(s, loc))))
               | (identifier - optional_schema - p.Suppress('=') - expression).setParseAction(pafac(TupleMemberNode))
               )
tuple_members = listMembers(';', tuple_member, pafac(TupleNode))
tuple = bracketedList('{', '}', ';', tuple_member, pafac(TupleNode))

# Variable (can't be any of the keywords, which may have lower matching priority)
variable = ~p.Or([p.Keyword(k) for k in keywords]) + identifier.copy().setParseAction(mkVar)

# Argument list will live by itself as a atom. Actually, it's a tuple, but we
# don't call it that because we use that term for something else already :)
arg_list = bracketedList('(', ')', ',', expression, ArgList)

parenthesized_expr = (sym('(') - expression - ')').setParseAction(head)

unary_op = (p.oneOf(' '.join(functions.unary_operators.keys())) - expression).setParseAction(mkUnOp)

if_then_else = (kw('if') + expression +
                kw('then') - expression -
                kw('else') - expression).setParseAction(doapply(Condition))

list_comprehension = (sym('[') + expression + kw('for') - variable - kw('in') -
    expression - p.Optional(kw('if') - expression) - sym(']')).setParseAction(doapply(ListComprehension))


# We don't allow space-application here
# Now our grammar is becoming very dirty and hackish
deref = p.Forward()
include = (kw('include') - deref).setParseAction(doapply(Include))

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
applic1 = (atom - p.ZeroOrMore(arg_list)).setParseAction(mkApplications)

# Dereferencing of an expression (obj.bar)
deref << (applic1 - p.ZeroOrMore(p.Suppress('.') - identifier)).setParseAction(mkDerefs)

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
applic2 = (term - p.ZeroOrMore(term)).setParseAction(mkApplications)

term = applic2
for op_level in functions.binary_operators_after_juxtaposition:
  operator_syms = ' '.join(op_level.keys())
  term = (term - p.ZeroOrMore(p.oneOf(operator_syms) - term)).setParseAction(mkBinOps)

expression << term

# Two entry points: start at an arbitrary expression, or expect the top-level
# scope to be a tuple.
start = expression.ignore(comment)
start_tuple = tuple_members.ignore(comment)


def reads(s, filename, loader, implicit_tuple):
  """Load but don't evaluate a GCL expression from a string."""
  try:
    the_context.filename = filename
    the_context.loader = loader
    return (start_tuple if implicit_tuple else start).parseWithTabs().parseString(s, parseAll=True)[0]
  except (p.ParseException, p.ParseSyntaxException) as e:
    msg = '%s:%d: %s\n%s\n%s^-- here' % (the_context.filename, e.lineno, e.msg, e.line, ' ' * (e.col - 1))
    raise exceptions.ParseError(msg)

