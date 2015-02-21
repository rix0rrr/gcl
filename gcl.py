"""
GCL -- Generic Configuration Language

GCL is built around named tuples, written with curly braces like this:

  {
    number = 1;
    string =  'value';
    bool =  true;
    expression = number * 2;
    list = [ 1, 2, 3 ];
  }

Semicolons are considered separators, so may be ommitted after the last
statement.

GCL has an expression language. Juxtaposition of two expressions (A B) means
tuple application, which returns a new tuple combining the keys of A and B,
such that B overwrites the keys of A already present.

This looks especially convenient when A is a reference and B is a tuple
literal:

  {
    foo_app = {
      program = 'foo';
      cwd = '/tmp';
    }

    my_foo = foo_app {
      cwd = '/home';
    }
  }

This makes it possible to do abstraction and parameterization (just define
tuples with the common components). To require derivations of a tuple to
fill in certain parameters, declare them without a value:

  {
    greet = {
      greeting;
      message = greeting + ' world';
    };

    hello_world = greet { message='hello' }
  }

If 'message' is evaluated, but greeting happens to not be filled in, an error
will be thrown. Expressions are lazily evaluated, so if 'message' is never
evaluated, this error will never be thrown. To force eager evaluation, use
complete() on a tuple.

Periods are used to dereference tuples:

  {
    tuple = {
      foo = 3;
    }

    that_foo = tuple.foo;
  }

Lines starting with # are comments.

Files can be included with 'include()', which takes a path relative to the file
the directive appears in (only in file context).

Notes:

  - Application is left-associative.
  - Function application of one argument allows omission of the parens like
    tuple application.

"""

import pyparsing as p


def do(*fns):
  def fg(args):
    for fn in fns:
      args = fn(args)
    return args
  return fg

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

#----------------------------------------------------------------------
#  Model
#

class EmptyEnvironment(object):
  def __getitem__(self, key):
    raise LookupError('Unbound variable: %r' % key)


class SourceLocation(object):
  def __init__(self, string, offset):
    self.string = string
    self.offset = offset


class Environment(object):
  """Binding environment, inherits from another Environment."""

  def __init__(self, values, parent=None):
    self.parent = parent or EmptyEnvironment()
    self.values = values

  def __getitem__(self, key):
    if key in self.values:
      return self.values[key]
    return self.parent[key]


class Thunk(object):
  def eval(self, env):
    raise NotImplementedError('Whoops')


class Void(Thunk):
  """An missing value."""
  def __init__(self):
    pass

  def eval(self, env):
    raise ValueError('Missing value')

  def __repr__(self):
    return '<unbound>'


class Constant(Thunk):
  """A GCL constant expression."""
  def __init__(self, value):
    self.value = value

  def eval(self, env):
    return self.value

  def __repr__(self):
    if self.value == True:
      return 'true'
    if self.value == False:
      return 'false'
    return repr(self.value)


class Var(Thunk):
  """Reference to another value."""
  def __init__(self, name, location):
    self.name = name
    self.location = location

  def eval(self, env):
    return env[self.name]

  def __repr__(self):
    return self.name


def mkVar(s, loc, toks):
  return Var(toks[0], SourceLocation(s, loc))


class List(Thunk):
  """A GCL list (either bound or unbound)."""
  def __init__(self, values):
    self.values = values

  def eval(self, env):
    return [v.eval(env) for v in self.values]

  def __repr__(self):
    return repr(self.values)


class UnboundTuple(Thunk):
  """Unbound tuple.

  When evaluating, the tuple doesn't actually evaluate its children. Instead,
  we return a (lazy) Tuple object that only evaluates the elements when they're
  requested.
  """
  def __init__(self, kv_pairs):
    self.items = dict(kv_pairs)

  def eval(self, env):
    return Tuple(self.items, env)

  def __repr__(self):
    return ('{' +
            '; '.join('%s = %r' % (key, value) for key, value in self.items.items()) +
            '}')


class Tuple(object):
  """Bound tuple, with lazy evaluation.

  Contains real values or Thunks. Thunks will be evaluated upon request, but
  not before.

  The parent_env is the environment in which we do lookups for values that are
  not in this Tuple.

  For a plain Tuple, this MUST be the global environment (because we don't want
  child tuples to implicitly inherit all values of all parent
  scopes--Borgconfig did that and it was horrible).

  For composited tuples, this will be the left tuple's environment.
  """
  def __init__(self, items, parent_env):
    self.__items = items
    self.__parent_env = parent_env

  def dict(self):
    return self.__items

  def __getitem__(self, key):
    try:
      x = self.__items[key]

      # Check if this is a Thunk that needs to be lazily evaluated before we
      # return it.
      if not isinstance(x, Thunk):
        return x

      if isinstance(x, UnboundTuple):
        # We make an exception if the Thunk is a tuple. In that case, we
        # don't evaluate in OUR environment (to prevent the tuple from
        # inheriting variables it's not supposed to). Instead, the tuple will
        # evaluate in our PARENT environment (which should be the global
        # environment).
        return x.eval(self.__parent_env)

      return x.eval(self.env())
    except Exception, e:
      raise LookupError("Can't get value for %r: %s" % (key, e))

  def __contains__(self, key):
    return key in self.__items

  def env(self):
    return Environment(self, self.__parent_env)

  def keys(self):
    return self.__items.keys()

  def items(self):
    return [(k, self[k]) for k in self.keys()]


class Application(Thunk):
  """Function application."""
  def __init__(self, functor, args):
    self.functor = functor
    self.args = args

  def eval(self, env):
    fn = self.functor.eval(env)
    args = [a.eval(env) for a in self.args]

    if isinstance(fn, Tuple):
      # Handle tuple application
      if len(args) != 1 or not isinstance(args[0], Tuple):
        raise ValueError('Tuple (%r) must be applied to exactly one other tuple (got %r)' %
                         (self.functor, self.args))

    # Any other callable type
    if not callable(fn):
      raise ValueError('Result of %r (%r) not callable' % (self.functor, fn))

    return fn(*args)


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


class Operation(Thunk):
  def __init__(self, op, operands):
    self.op = op
    self.operands = operands

  def eval(self, env):
    ops = [x.eval(env) for x in self.operands]
    ret = ops[0]
    for x in ops[1:]:
      ret = self.combine(ret, x)
    return ret

  def combine(self, a, b):
    if self.op == '*':
      return a * b
    if self.op == '/':
      return a / b
    if self.op == '+':
      return a + b
    if self.op == '-':
      return a - b

    raise LookupError('Unknown operator: %s' % self.op)


  def __repr__(self):
    return (' %s ' % self.op).join(repr(x) for x in self.operands)

def mkOperation(op):
  def combiner(tokens):
    if len(tokens) == 1:
      return tokens[0]
    return Operation(op, tokens)
  return combiner

#----------------------------------------------------------------------
#  Grammar
#

def bracketedList(l, r, sep, expr, what):
  """Parse bracketed list.

  Empty list is possible, as is a trailing separator.
  """
  return (p.Literal(l) + p.Optional(
    p.delimitedList(expr, sep) +
    p.Optional(sep).suppress()) +
    p.Literal(r)).setParseAction(do(inner, what))

expression = p.Forward()

comment = '#' + p.restOfLine

identifier = p.Word(p.alphanums + '_')

# Contants
integer = p.Combine(p.Optional('-') + p.Word(p.nums)).setParseAction(do(head, int, Constant))
floating = p.Combine(p.Optional('-') + p.Optional(p.Word(p.nums)) + '.' + p.Word(p.nums)).setParseAction(do(head, float, Constant))
dq_string = p.QuotedString('"', escChar='\\', multiline=True).setParseAction(do(head, Constant))
sq_string = p.QuotedString("'", escChar='\\', multiline=True).setParseAction(do(head, Constant))
boolean = p.Or(['true', 'false']).setParseAction(do(head, mkBool, Constant))

# List
list_ = bracketedList('[', ']', ',', expression, List)

# Tuple
tuple_member = ((identifier + '=' + expression).setParseAction(lambda x: (x[0], x[2]))
               | (identifier + ~p.FollowedBy('=')).setParseAction(lambda x: (x[0], Void())))
tuple = bracketedList('{', '}', ';', tuple_member, UnboundTuple)

# Variable
variable = identifier.copy().setParseAction(mkVar)

# Argument list will live by itself as a atom. Actually, it's a tuple, but we
# don't call it that because we use that term for something else already :)
arg_list = bracketedList('(', ')', ',', expression, list)

atom = (floating
        | integer
        | dq_string
        | sq_string
        | boolean
        | list_
        | arg_list
        | tuple
        | variable
        )

# All application is juxtaposition, but sometimes it's juxtaposition against an
# unnamed tuple (arglist): foo (1, 2, 3).
application = (atom + p.ZeroOrMore(p.Group(atom))).setParseAction(mkApplications)

term = (application + p.ZeroOrMore(p.Literal('*').suppress() + application)).setParseAction(mkOperation('*'))

expr = (term + p.ZeroOrMore(p.Literal('+').suppress() + term)).setParseAction(mkOperation('+'))

expression << expr

start = expression.ignore(comment)

# Notes: operator precedence
# Includes must be w.r.t. the INCLUDING file name (pass in thru global?)

#----------------------------------------------------------------------
#  Top-level functions
#

def loads(s):
  """Load a GCL expression from a string."""
  return start.parseString(s, parseAll=True)[0]
