"""
AST and parsing related functions.
"""
import collections
import functools
import itertools
import logging
import textwrap

import sparse

from . import framework


logger = logging.getLogger(__name__)

class ParseContext(object):
  def __init__(self):
    self.filename = '<from string>'
    self.loader = None
the_context = ParseContext()

#----------------------------------------------------------------------
#  AST NODES

class AstNode(object):
  """Base class for AST nodes.

  All AST nodes have:

  - An identity, for cached evaluation.
  - Child nodes which it contains.
  """
  def __init__(self, *child_nodes):
    self.child_nodes = child_nodes
    print self.child_nodes
    self.span = sum((c.span for c in self.child_nodes), sparse.empty_span)
    self.returned_by_find_query = True

  def find_tokens(self, q, into=None):
    """Find all AST nodes that encompass the given source location.

    Filename, line and column."""
    into = into if into is not None else []

    if self.span.contains(q) and self.returned_by_find_query:
      into.append(self)

    for child in self.child_nodes:
      child.find_tokens(q, into)

    return into

  def reraise_in_context(self, e, doing='while evaluating'):
    lines = self.span.annotated_source(doing)
    msg = '\n'.join(lines)
    raise exceptions.EvaluationError(msg, e)


class Var(framework.Thunk, AstNode):
  """Variable reference."""

  def __init__(self, identifier_token):
    framework.Thunk.__init__(self)
    AstNode.__init__(self, identifier_token)

    self.ident = framework.obj_ident()
    self.identifier_token = identifier_token
    self.identifier = identifier_token.value

  def eval(self, env):
    try:
      return env[self.identifier]
    except exceptions.EvaluationError as e:
      self.reraise_in_context(e)

  def __repr__(self):
    return self.identifier


class List(framework.Thunk, AstNode):
  """List literal."""
  def __init__(self, *elements):
    # FIXME: span Demarcation of lists is not correct! We're missing the
    # suppressed tokens!
    framework.Thunk.__init__(self)
    AstNode.__init__(self, *elemenst)

    self.elements = list(elements)

  def eval(self, env):
    return [framework.eval(v, env) for v in self.elements]

  def __repr__(self):
    return repr(self.elements)


class ArgList(framework.Thunk, AstNode):
  """A parenthesized, comma-separated argument list.

  This is actually a shallow wrapper for Python's list type. We can't use that
  because pyparsing will automatically concatenate lists, which we don't want
  in this case.
  """
  def __init__(self, *elements):
    framework.Thunk.__init__(self)
    AstNode.__init__(self, *elements)

    self.elements = elements

    # Not a user-visible construct, return the elements but not the argument list
    self.returned_by_find_query = False

  def eval(self, env):
    return [framework.eval(v, env) for v in self.elements]

  def __repr__(self):
    return '(%s)' % ', '.join(repr(x) for x in self.elements)


class TupleMemberNode(AstNode):
  """AST node for tuple members.

  They have a name, a comment, an expression value and an optional schema.
  """
  def __init__(self, identifier_token, schema_node, value_expression):
    AstNode.__init__(self, identifier_token, schema_node, value_expression)

    self.identifier_token = identifier_token
    self.name             = identifier_token.value
    self.schema_node      = schema_node
    self.value_expression = value_expression
    self.comment          = DocComment()

    # Hack: set the name on the value if it is a Void value, to help the Void
    # generate better error messages. Not doing it here makes the grammar a lot
    # messier.
    if isinstance(self.value_expression, Void):
      self.value_expression.name = self.name

  def attach_comment(self, comment):
    self.comment = comment

  def __repr__(self):
    schema_repr = ': %r' % self.schema_node if not isinstance(self.schema_node, NoSchemaNode) else ''
    return '%s%s = %r' % (self.name, schema_repr, self.value_expression)


class TupleNode(framework.Thunk, AstNode):
  """AST node for tuple

  When evaluating, the tuple doesn't actually evaluate its children. Instead, we return a (lazy)
  Tuple object that only evaluates the elements when they're requested.
  """
  def __init__(self, allow_errors, *members):
    framework.Thunk.__init__(self)
    AstNode.__init__(self, *members)

    # Filter contents down to actual members (ignore UnparseableNodes)
    members = sorted((m for m in members if is_tuple_member(m)), key=lambda m: m.name)
    for name, members in itertools.groupby(members, key=lambda m: m.name):
      members = list(members)  # Reify iterator
      if len(members) > 1 and not allow_errors:
        raise exceptions.ParseError(members[1].identifier_token.span,
                                    'Key %s occurs more than once in tuple' % name)

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
    self.members = [TupleMemberNode(sparse.Token('identifier', key, sparse.empty_span))
                    for key, value in dct.items()]
    self.member = {m.name: m for m in self.members}


def dict2tuple(dct):
    return runtime.Tuple(RuntimeTupleNode(dct), framework.EmptyEnvironment(), dict2tuple)


class DocComment(object):
  def __init__(self, *lines):
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

  def body(self):
    return '\n'.join(self.body_lines())

  def as_string(self):
    title = self.title()
    body = '\n'.join(self.body_lines())

    return (title + '\n\n' + body) if body else title

  def __repr__(self):
    return '\n'.join('#. %s' % l for l in self.lines)


class Void(framework.Thunk, AstNode):
  """A missing value."""
  def __init__(self):
    framework.Thunk.__init__(self)
    AstNode.__init__(self)

    self.void_name = None

  def eval(self, env):
    raise exceptions.EvaluationError('Unbound value')

  def is_unbound(self):
    # Overridden from Thunk
    return True

  def __repr__(self):
    return '<unbound>'


class Inherit(framework.BindableThunk, AstNode):
  """Inherit Thunks can be either bound or unbound."""
  def __init__(self, identifier_token, env=None):
    framework.BindableThunk.__init__(self)
    AstNode.__init__(self, identifier_token)

    self.identifier_token = identifier_token
    self.identifier = identifier_token.value
    self.env = env

  def bind(self, env):
    return Inherit(self.identifier_token, env)

  def eval(self, env):
    if not self.env:
      raise exceptions.EvaluationError("Inherited key")
    return self.env[self.name]

  def __repr__(self):
    return 'inherit %s' % self.identifier

  @staticmethod
  def make(*tokens):
    return [TupleMemberNode(token, no_schema, Inherit(token))
            for token in tokens]


class UnOp(framework.Thunk, AstNode):
  def __init__(self, op_token, right):
    framework.Thunk.__init__(self)
    AstNode.__init__(self, op_token, right)

    self.op = op_token.value
    self.right = right

    self.returned_by_find_query = False

  def eval(self, env):
    fn = functions.unary_operators.get(self.op, None)
    if fn is None:
      raise exceptions.EvaluationError('Unknown unary operator: %s' % self.op)
    return call_fn(fn, ArgList([self.right]), env)

  def __repr__(self):
    return '%s%s%r' % (self.op, ' ' if self.op == 'not' else '', self.right)

  @staticmethod
  def make(*tokens):
    return UnOp(tokens[0], tokens[1])


class BinOp(framework.Thunk, AstNode):
  def __init__(self, left, op_token, right):
    framework.Thunk.__init__(self)
    AstNode.__init__(self, left, op_token, right)

    self.left = left
    self.op = op.value
    self.right = right

    self.returned_by_find_query = False

  def eval(self, env):
    fn = functions.all_binary_operators.get(self.op, None)
    if fn is None:
      raise exceptions.EvaluationError('Unknown operator: %s' % self.op)

    return call_fn(fn, ArgList([self.left, self.right]), env)

  def __repr__(self):
    return ('%r %s %r' % (self.left, self.op, self.right))

  @staticmethod
  def make(*tokens):
    tokens = list(tokens)
    while len(tokens) > 1:
      assert(len(tokens) >= 3)
      tokens[0:3] = [BinOp(tokens[0], tokens[1], tokens[2])]
    return tokens[0]


class Condition(framework.Thunk, AstNode):
  def __init__(self, cond, then, else_):
    framework.Thunk.__init__(self)
    AstNode.__init__(self, cond_, then, else_)

    self.cond = cond
    self.then = then
    self.else_ = else_

    self.returned_by_find_query = False

  def eval(self, env):
    if framework.eval(self.cond, env):
      return framework.eval(self.then, env)
    else:
      return framework.eval(self.else_, env)

  def __repr__(self):
    return 'if %r then %r else %r' % (self.cond, self.then, self.else_)


class ListComprehension(framework.Thunk, AstNode):
  def __init__(self, expr, var_token, collection, cond=None):
    framework.Thunk.__init__(self)
    AstNode.__init__(self, expr, var_token, collection, cond)

    self.expr = expr
    self.var = var_token.value
    self.collection = collection
    self.cond = cond

  def eval(self, env):
    ret = []
    for x in framework.eval(self.collection, env):
      new_env = framework.Environment({self.var:x}, env)
      if self.cond is None or framework.eval(self.cond, new_env):
        ret.append(framework.eval(self.expr, new_env))
    return ret

  def __repr__(self):
    return '[%r for %s in %r]' % (self.expr, self.var, self.collection)


class Include(framework.Thunk, AstNode):
  def __init__(self, file_ref):
    framework.Thunk.__init__(self)
    AstNode.__init__(self, file_ref)

    self.file_ref = file_ref
    self.current_file = file_ref.span.file.filename
    self.loader = the_context.loader

  def eval(self, env):
    file_ref = framework.eval(self.file_ref, env)
    if not framework.is_str(file_ref):
      raise exceptions.EvaluationError('Included argument (%r) must be a string, got %r' %
                            (self.file_ref, file_ref))

    loaded = self.loader(self.current_file, file_ref, env=env.root)
    assert not isinstance(loaded, framework.Thunk)
    return loaded

  def __repr__(self):
    return 'include(%r)' % self.file_ref


class Literal(framework.Thunk, AstNode):
  """A GCL literal expression."""
  def __init__(self, value):
    framework.Thunk.__init__(self)
    AstNode.__init__(self, value)

    self.value = value

  def eval(self, env):
    return self.value

  def __repr__(self):
    if type(self.value) == bool:
      return 'true' if self.value else 'false'
    return repr(self.value)


class Application(framework.Thunk, AstNode):
  """Function application."""
  def __init__(self, left, right):
    framework.Thunk.__init__(self)
    AstNode.__init__(self, value)

    self.left = left
    self.right = right

  def right_as_list(self):
    return self.right if isinstance(self.right, ArgList) else ArgList([self.right])

  def eval_right_as_list(self, env):
    return framework.eval(self.right_as_list(), env)

  def eval(self, env):
    with framework.EvaluationContext(validate=False):
      # Temporarily disable validation
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
      self.reraise_in_context(e, 'while calling \'%r\'' % self)

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

  @staticmethod
  def make(*atoms):
    """Make a sequence of applications from a list of tokens.

    atoms is a list of atoms, which will be handled left-associatively. E.g:

        ['foo', [], []] == foo()() ==> Application(Application('foo', []), [])
    """
    atoms = list(atoms)
    while len(atoms) > 1:
      atoms[0:2] = [Application(atoms[0], atoms[1])]

    # Nothing left to apply
    return atoms[0]


class Deref(framework.Thunk, AstNode):
  """Dereferencing of a dictionary-like object."""
  def __init__(self, haystack, needle):
    framework.Thunk.__init__(self)
    AstNode.__init__(self, haystack, needle)

    self._haystack = haystack
    self.needle = needle

  def _children(self):
    return [self._haystack, self.needle]

  def haystack(self, env):
    return framework.eval(self._haystack, env)

  def eval(self, env):
    try:
      return self.haystack(env)[self.needle.name]
    except exceptions.EvaluationError as e:
      self.reraise_in_context(e, 'while evaluating \'%r\'' % self)
    except TypeError as e:
      self.reraise_in_context(e, 'while getting %r from %r' % (self.needle, self._haystack))

  def __repr__(self):
    return '%s.%s' % (self._haystack, self.needle)

  @staticmethod
  def make(*tokens):
    tokens = list(tokens)
    while len(tokens) > 1:
      tokens[0:2] = [Deref(tokens[0], tokens[1])]
    return tokens[0]


def nonempty(s):
  """Return True iff the given string is nonempty."""
  return len(s.strip())


def drop(n, xs):
  for i, x in enumerate(xs):
    if n <= i:
      yield x


def attach_doc_comment(*args):
  comments, member = args[:-1], args[-1]

  for comment in comments:
    member.attach_comment(comment)

  return member


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


def is_tuple_member(x):
  return isinstance(x, TupleMemberNode)


#----------------------------------------------------------------------
#  Schema AST model
#

class NoSchemaNode(framework.Thunk, AstNode):
  """For values without a schema."""
  def __init__(self):
    framework.Thunk.__init__(self)
    AstNode.__init__(self)

    self.required = False
    self.private = False

  def eval(self, env):
    return schema.AnySchema()

no_schema = NoSchemaNode()  # Singleton object


class AnySchemaExprNode(framework.Thunk, AstNode):
  def __init__(self):
    framework.Thunk.__init__(self)
    AstNode.__init__(self)

  def eval(self, env):
    return schema.AnySchema()

any_schema_expr = AnySchemaExprNode()


class MemberSchemaNode(framework.Thunk, AstNode):
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
  def __init__(self, private, required, expr):
    framework.Thunk.__init__(self)
    AstNode.__init__(self, expr)

    # FIXME: Private and required are also tokens (or should be)
    self.private = private
    self.required = required
    self.expr = expr

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
#  PARSING

# Scanning for all of these at once is faster than scanning for them individually.
keywords = ['inherit', 'if', 'then', 'else', 'include', 'null', 'for', 'in', 'private', 'required']

scanner = sparse.Scanner([
    sparse.WHITESPACE,
    sparse.Syntax('comment', '#$|#[^.].*$'),
    sparse.Syntax('doc_comment', '#\.(.*)$', lambda s: s[2:].strip()),
    # Keywords
    sparse.Syntax('bool_op', r'\band\b|\bor\b'),
    sparse.Syntax('minus', r'-(?!\d)'),
    sparse.Syntax('not', r'\bnot\b'),
    sparse.Syntax('keyword', '|'.join(r'\b' + k + r'\b' for k in keywords)),
    sparse.Syntax('bool_literal', r'\btrue\b|\bfalse\b', bool),
    # Identifiers (must come after keywords for matching priority)
    sparse.Syntax('identifier', sparse.quoted_string_regex('`'), sparse.quoted_string_process),
    sparse.Syntax('identifier', r'[a-zA-Z_]([a-zA-Z0-9_:-]*[a-zA-Z0-9_])?'),
    # Other atoms
    sparse.Syntax('string_literal', sparse.quoted_string_regex('"'), sparse.quoted_string_process),
    sparse.Syntax('string_literal', sparse.quoted_string_regex("'"), sparse.quoted_string_process),
    sparse.Syntax('compare_op', '|'.join(['<=', '>=', '==', '!=', '<', '>'])),
    sparse.Syntax('mul_op', '[*/%]'),
    sparse.Syntax('plus', '\+'),
    sparse.Syntax('float_literal', r'-?\d+\.\d+', float),
    sparse.Syntax('int_literal', r'-?\d+', int),
    # Symbols
    sparse.Syntax('symbol', '[' + ''.join('\\' + s for s in '[](){}=,;:.') + ']'),
    ])


def listMembers(t_sep, expr):
  return sparse.delimited_list(expr, sparse.Q(t_sep))


def bracketedList(t_l, t_r, t_sep, expr, allow_missing_close=False):
  """Parse bracketed list.

  Empty list is possible, as is a trailing separator.
  """
  closer = sparse.Q(t_r) if not allow_missing_close else sparse.Optional(sparse.Q(t_r))
  return sparse.Q(t_l) - listMembers(t_sep, expr) + closer


GRAMMAR_CACHE = {}
def make_grammar(allow_errors):
  """Make the part of the grammar that depends on whether we swallow errors or not."""
  T = sparse.T
  Q = sparse.Q
  p = sparse
  Rule = sparse.Rule

  if allow_errors in GRAMMAR_CACHE:
    return GRAMMAR_CACHE[allow_errors]

  class Grammar:
    expression = p.Forward()

    variable = Rule('variable') >> T('identifier') >> Var

    # Lists (these need to be backtrackable because they're ambiguous at the left)
    list_ = Rule('list') >> Q('[') + sparse.delimited_list(expression, Q(',')) + Q(']') >> List
    list_comprehension = Rule('list_comprehension') >> (
        Q('[') + expression + Q('for') - T('identifier') - Q('in') - expression + p.Optional(Q('if') - expression) + Q(']')
        ) >> ListComprehension

    # Tuple
    inherit_member = Rule('inherit') >> Q('inherit') - p.ZeroOrMore(T('identifier')) >> Inherit.make
    schema_spec = Rule('schema_spec') >> (
        p.Optional(T('private').action(lambda _: True), default=False)
        + p.Optional(T('required').action(lambda _: True), default=False)
        + p.Optional(expression)
        ) >> MemberSchemaNode
    optional_schema   = Rule('optional_schema')   >> p.Optional(Q(':') - schema_spec, default=no_schema)

    expression_value  = Rule('expression_value')  >> Q('=') - expression
    member_value      = Rule('member_value')      >> p.Optional(expression_value, default=Void())
    named_member      = Rule('named_member')      >> T('identifier') - optional_schema + member_value >> TupleMemberNode
    documented_member = Rule('documented_member') >> p.ZeroOrMore(T('doc_comment')) + named_member >> attach_doc_comment
    tuple_member      = Rule('tuple_member')      >> inherit_member | documented_member

    ErrorAwareTupleNode = functools.partial(TupleNode, allow_errors)
    tuple_members       = Rule('tuple_members') >> listMembers(';', tuple_member) >> ErrorAwareTupleNode
    tuple               = Rule('tuple')         >> Q('{') - tuple_members + Q('}') >> ErrorAwareTupleNode

    # Argument list will live by itself as a atom. Actually, it's a tuple, but we
    # don't call it that because we use that term for something else already :)
    arg_list = Rule('arg_list') >> bracketedList('(', ')', ',', expression) >> ArgList

    parenthesized_expr = Rule('parenthesized_expr') >> p.parenthesized(expression)

    unary_op = Rule('unary_op') >> (T('minus') | T('plus') | T('not')) - expression >> UnOp.make

    if_then_else = Rule('if_then_else') >> Q('if') - expression + Q('then') - expression + Q('else') - expression >> Condition

    # We don't allow space-application here
    # Now our grammar is becoming very dirty and hackish
    deref = p.Forward()
    include = Rule('include') >> Q('include') - deref >> Include

    literal = (T('string_literal')
            | T('float_literal')
            | T('int_literal')
            | T('bool_literal')
            | T('null').action(lambda _: None)
            ).action(Literal)

    atom = Rule('atom') >> (tuple
            | literal
            | variable
            | list_comprehension
            | list_
            | unary_op
            | parenthesized_expr
            | if_then_else
            | include
            )

    # We have two different forms of function application, so they can have 2
    # different precedences. This one: fn(args), which binds stronger than
    # dereferencing (fn(args).attr == (fn(args)).attr)
    applic1 = Rule('applic1') >> atom - p.ZeroOrMore(arg_list) >> Application.make

    # Dereferencing of an expression (obj.bar)
    deref.set(Rule('deref') >> applic1 - p.ZeroOrMore(Q('.') - T('identifier')) >> Deref.make)

    # Binary operators before juxtaposition
    factor = deref
    term = Rule() >> factor - p.ZeroOrMore((T('mul_op') | T('in')) - factor) >> BinOp.make
    applicable = Rule() >> term - p.ZeroOrMore(T('add_op') - term) >> BinOp.make

    # Juxtaposition
    juxtaposed = Rule('juxtaposed') >> applicable - p.ZeroOrMore(applicable) >> Application.make

    # Binary operators after juxtaposition
    compared = Rule('compared') >> juxtaposed - p.ZeroOrMore(T('compare_op') - juxtaposed) >> BinOp.make
    expression.set(Rule('expression') >> compared - p.ZeroOrMore(T('bool_op') - compared) >> BinOp.make)

    # Two entry points: start at an arbitrary expression, or expect the top-level
    # scope to be a tuple.
    start = expression
    start_tuple = tuple_members
  GRAMMAR_CACHE[allow_errors] = Grammar
  return Grammar


def reads(s, filename, loader, implicit_tuple, allow_errors):
  """Load but don't evaluate a GCL expression from a string."""
  try:
    the_context.filename = filename
    the_context.loader = loader

    file = sparse.File(filename, s)
    tokens = scanner.tokenize(file)

    grammar = make_grammar(allow_errors=allow_errors)
    root = grammar.start_tuple if implicit_tuple else grammar.start
    parser = sparse.make_parser(root)
    result = sparse.parse_all(parser, tokens, file)

    #import sys
    #sparse.print_parser(parser, sys.stdout)

    return result[0]
  except sparse.ParseError as e:
    raise e.add_context(filename, s)


if __name__ == '__main__':
    grammar = make_grammar(allow_errors=False)
    parser = sparse.make_parser(grammar.deref)
