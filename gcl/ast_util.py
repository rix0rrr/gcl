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
  for node in ast_rootpath[1:]:
    if is_tuple_node(node):
      inflated = inflated[node.name]
  return inflated


def is_tuple_node(x):
  return isinstance(x, ast.TupleNode)


def is_tuple_member_node(x):
  return isinstance(x, ast.TupleMemberNode)


def is_deref_node(x):
  return isinstance(x, ast.Deref)


def enumerate_scope(ast_rootpath, include_default_builtins=False):
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
    for k in gcl.default_env.keys():
      scope[k] = None

  return scope


def find_completions(ast_rootpath, root_env=gcl.default_env):
  if not ast_rootpath:
    return []
  tup = inflate_context_tuple(ast_rootpath, root_env)
  path = path_until(ast_rootpath, is_deref_node)
  if not path:
    return []
  deref = path[-1]
  haystack = deref.haystack(tup.env(tup))
  return haystack.keys()
