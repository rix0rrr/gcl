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


def inflate_tuple(ast_rootpath, root_env):
  """Instantiate a Tuple from a TupleNode.

  Walking the AST tree upwards, evaluate from the root down again.
  """
  # We only need to look at tuple members going down.
  inflated = ast_rootpath[0].eval(root_env)
  for node in ast_rootpath[1:]:
    if is_tuple_member_node(node):
      inflated = inflated[node.name]
  return inflated


def is_tuple_node(x):
  return isinstance(x, ast.TupleNode)


def is_tuple_member_node(x):
  return isinstance(x, ast.TupleMemberNode)


# def enumerate_scope(ast_rootpath, root_env):
  # """Return a list of (name, node) pairs for the given tuple node.

  # Enumerates all keys that are in scope in a given tupe. The node
  # part of the tuple may be None, in case the binding is a built-in.
  # """
  # tup = inflate_tuple(path_until(ast_rootpath, is_tuple_node), root_env)
  # env = tup.env(tup)
  # for key in env.keys():
    # yield key, env.get_node(key)

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
