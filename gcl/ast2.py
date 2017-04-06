"""
AST and parsing related functions.
"""
import functools
import logging

import sparse


logger = logging.getLogger(__name__)

class ParseContext(object):
  def __init__(self):
    self.filename = '<from string>'
    self.loader = None
the_context = ParseContext()

#----------------------------------------------------------------------
#  AST NODES

def named_coll(klassname):
  class Result(object):
    def __init__(self, *args):
      self.args = args

    def __repr__(self):
      return '%s%r' % (klassname, self.args)
  return Result


Var = named_coll('Var')
List = named_coll('List')
inheritNodes = named_coll('Inherit')
MemberSchemaNode = named_coll('MemberSchemaNode')
Void = named_coll('Void')
TupleNode = named_coll('TupleNode')
TupleMemberNode = named_coll('TupleMemberNode')
ArgList = named_coll('ArgList')
attach_doc_comment = named_coll('AttachDocComment')
mkUnOp = named_coll('mkUnOp')
mkBinOps = named_coll('mkBinOps')
Condition = named_coll('Condition')
ListComprehension = named_coll('ListComprehension')
Include = named_coll('Include')
Literal = named_coll('Literal')
mkApplications = named_coll('Applications')
mkDerefs = named_coll('Derefs')

#----------------------------------------------------------------------
#  PARSING

# Scanning for all of these at once is faster than scanning for them individually.
keywords = ['inherit', 'if', 'then', 'else', 'include', 'null', 'for', 'in', 'private', 'required']

scanner = sparse.Scanner([
    sparse.WHITESPACE,
    sparse.Syntax('comment', '#$|#[^.].*$'),
    sparse.Syntax('doc_comment', '#\.(.*)$', lambda s: s[2:].strip()),
    # Keywords
    sparse.Syntax('bool_op', 'and|or'),
    sparse.Syntax('minus', '-'),
    sparse.Syntax('not', 'not'),
    sparse.Syntax('keyword', '|'.join(k for k in keywords)),
    sparse.Syntax('bool_literal', 'true|false', bool),
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
    inherit_member = Rule('inherit') >> Q('inherit') - p.ZeroOrMore(T('identifier')) >> inheritNodes
    schema_spec = Rule('schema_spec') >> (
        p.Optional(T('private').action(lambda _: True), default=False)
        + p.Optional(T('required').action(lambda _: True), default=False)
        + p.Optional(expression)
        ) >> MemberSchemaNode
    optional_schema   = Rule('optional_schema')   >> p.Optional(Q(':') - schema_spec)

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

    unary_op = Rule('unary_op') >> (T('minus') | T('plus') | T('not')) >> mkUnOp

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
    applic1 = Rule('applic1') >> atom - p.ZeroOrMore(arg_list) >> mkApplications

    # Dereferencing of an expression (obj.bar)
    deref.set(Rule('deref') >> applic1 - p.ZeroOrMore(Q('.') - T('identifier')) >> mkDerefs)

    # Binary operators before juxtaposition
    factor = deref
    term = Rule() >> factor - p.ZeroOrMore((T('mul_op') | T('in')) - factor) >> mkBinOps
    applicable = Rule() >> term - p.ZeroOrMore(T('add_op') - term) >> mkBinOps

    # Juxtaposition
    juxtaposed = Rule('juxtaposed') >> applicable - p.ZeroOrMore(applicable) >> mkApplications

    # Binary operators after juxtaposition
    compared = Rule('compared') >> juxtaposed - p.ZeroOrMore(T('compare_op') - juxtaposed) >> mkBinOps
    expression.set(Rule('expression') >> compared - p.ZeroOrMore(T('bool_op') - compared) >> mkBinOps)

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

    tokens = scanner.tokenize(s)

    grammar = make_grammar(allow_errors=allow_errors)
    root = grammar.start_tuple if implicit_tuple else grammar.start
    parser = sparse.make_parser(root)
    import sys
    sparse.print_parser(parser, sys.stdout)
    result = sparse.parse_all(parser, tokens)

    return result[0]
  except sparse.ParseError as e:
    raise e.add_context(filename, s)


if __name__ == '__main__':
    grammar = make_grammar(allow_errors=False)
    parser = sparse.make_parser(grammar.deref)
