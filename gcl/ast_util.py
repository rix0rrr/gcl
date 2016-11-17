"""GCL ast utility functions.

These utility functions are intended for GCL tools (in particular, gcls).
"""
from __future__ import absolute_import

import gcl
from . import framework
from . import exceptions
from . import ast


def path_until(rootpath, pred):
  for i in range(len(rootpath), 0, -1):
    if pred(rootpath[i - 1]):
      return rootpath[:i]
  return []


def inflate_context_tuple(ast_rootpath, root_env):
  """Instantiate a Tuple from a TupleNode.

  Walking the AST tree upwards, evaluate from the root down again.
  """
  # We only need to look at tuple members going down.
  inflated = ast_rootpath[0].eval(root_env)
  try:
    for member, tuple in pair_iter(ast_rootpath[1:]):
      if is_tuple_member_node(member) and is_tuple_node(tuple):
        inflated = inflated[member.name]
  except gcl.EvaluationError:
    # Eat evaluation error, probably means the rightmost tuplemember wasn't complete.
    # Return what we have so far.
    pass
  return inflated


def is_tuple_node(x):
  return isinstance(x, ast.TupleNode)


def is_tuple_member_node(x):
  return isinstance(x, ast.TupleMemberNode)


def is_identifier(x):
  return isinstance(x, ast.Identifier)


def is_deref_node(x):
  return isinstance(x, ast.Deref)


def enumerate_scope(ast_rootpath, root_env=None, include_default_builtins=False):
  """Return a dict of { name => node } for the given tuple node.

  Enumerates all keys that are in scope in a given tuple. The node
  part of the tuple may be None, in case the binding is a built-in.
  """
  scope = {}
  for node in reversed(ast_rootpath):
    if is_tuple_node(node):
      for member in node.members:
        if member.name not in scope:
          scope[member.name] = member

  if include_default_builtins:
    root_env = gcl.default_env

  if root_env:
    for k in root_env.keys():
      scope[k] = None

  return scope


def find_completions(ast_rootpath, root_env=gcl.default_env):
  tup = inflate_context_tuple(ast_rootpath, root_env)
  path = path_until(ast_rootpath, is_deref_node)
  if not path:
    return []
  deref = path[-1]
  haystack = deref.haystack(tup.env(tup))
  return haystack.keys()


def find_completions_at_cursor(ast_tree, filename, line, col, root_env=gcl.default_env):
  """Find completions at the cursor.

  Return a dictionary of { name => builtin }
  """
  q = gcl.SourceQuery(filename, line, col - 1)
  rootpath = ast_tree.find_tokens(q)

  if len(rootpath) >= 2 and is_tuple_member_node(rootpath[-2]) and is_identifier(rootpath[-1]):
    # The cursor is in identifier-position in a member declaration. In that case, we
    # don't return any completions.
    return []

  derefs = find_completions(rootpath)
  if derefs:
    return {name: False for name in derefs}

  scope = enumerate_scope(rootpath, root_env=root_env)
  return {name: node is None for name, node in scope.items()}


def pair_iter(xs):
  last = None
  for x in xs:
    if last is not None:
      yield (last, x)
    last = x
