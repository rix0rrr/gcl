"""GCL ast utility functions.

These utility functions are intended for GCL tools (in particular, gcls).
"""
from __future__ import absolute_import

import collections
import textwrap

import gcl
from . import ast
from . import exceptions
from . import framework
from . import runtime
from . import schema
from . import util

def path_until(rootpath, pred):
  for i in range(len(rootpath), 0, -1):
    if pred(rootpath[i - 1]):
      return rootpath[:i]
  return []


def inflate_context_tuple(ast_rootpath, root_env):
  """Instantiate a Tuple from a TupleNode.

  Walking the AST tree upwards, evaluate from the root down again.
  """
  with util.LogTime('inflate_context_tuple'):
    # We only need to look at tuple members going down.
    inflated = ast_rootpath[0].eval(root_env)
    current = inflated
    env = root_env
    try:
      for node in ast_rootpath[1:]:
        if is_tuple_member_node(node):
          assert framework.is_tuple(current)
          with util.LogTime('into tuple'):
            thunk, env = inflated.get_thunk_env(node.name)
            current = framework.eval(thunk, env)

        elif framework.is_list(current):
          with util.LogTime('eval thing'):
            current = framework.eval(node, env)

        if framework.is_tuple(current):
          inflated = current
    except (gcl.EvaluationError, ast.UnparseableAccess):
      # Eat evaluation error, probably means the rightmost tuplemember wasn't complete.
      # Return what we have so far.
      pass
    return inflated


def is_tuple_node(x):
  return isinstance(x, ast.TupleNode)


def is_tuple_member_node(x):
  return isinstance(x, ast.TupleMemberNode)


def is_deref_node(x):
  return isinstance(x, ast.Deref)


def is_thunk(x):
  return isinstance(x, framework.Thunk)


def enumerate_scope(ast_rootpath, root_env=None, include_default_builtins=False):
  """Return a dict of { name => Completions } for the given tuple node.

  Enumerates all keys that are in scope in a given tuple. The node
  part of the tuple may be None, in case the binding is a built-in.
  """
  with util.LogTime('enumerate_scope'):
    scope = {}
    for node in reversed(ast_rootpath):
      if is_tuple_node(node):
        for member in node.members:
          if member.name not in scope:
            scope[member.name] = Completion(member.name, False, member.comment.as_string(), member.span)

    if include_default_builtins:  # Backwards compat flag
      root_env = gcl.default_env

    if root_env:
      for k in root_env.keys():
        if k not in scope and not hide_from_autocomplete(root_env[k]):
          v = root_env[k]
          scope[k] = Completion(k, True, dedent(v.__doc__ or ''), None)

    return scope


def hide_from_autocomplete(value):
  # This is a bit silly, but the default schema types are in the default environment
  # as well. Don't know why I decided that, but rather than refactor that, easier
  # to exclude them from the autocomplete list.
  return isinstance(value, schema.Schema)


def dedent(docstring):
  parts = textwrap.dedent(docstring or '').split('\n', 1)
  if len(parts) == 1:
    return parts[0]
  return parts[0] + '\n\n' + strip_initial_empty_line(textwrap.dedent(parts[1]))


def strip_initial_empty_line(x):
  lines = x.split('\n')
  if not lines[0].strip():
    return '\n'.join(lines[1:])
  return x


def find_deref_completions(ast_rootpath, root_env=gcl.default_env):
  """Returns a dict of { name => Completions }."""
  with util.LogTime('find_deref_completions'):
    tup = inflate_context_tuple(ast_rootpath, root_env)
    path = path_until(ast_rootpath, is_deref_node)
    if not path:
      return {}
    deref = path[-1]
    haystack = deref.haystack(tup.env(tup))
    if not hasattr(haystack, 'keys'):
      return {}
    return {n: get_completion(haystack, n) for n in haystack.keys()}


def get_completion(haystack, name):
  thunk = haystack.get_member_node(name)
  return Completion(name, False, thunk.comment.as_string(), thunk.span)


def is_identifier_position(rootpath):
  """Return whether the cursor is in an identifier declaration."""
  return rootpath and isinstance(rootpath[-1], ast.IntroIdentifier)


def find_completions_at_cursor(ast_tree, filename, line, col, root_env=gcl.default_env):
  """Find completions at the cursor.

  Return a dict of { name => Completion } objects.
  """
  q = gcl.SourceQuery(filename, line, col - 1)
  rootpath = ast_tree.find_tokens(q)

  if is_identifier_position(rootpath):
    return find_inherited_key_completions(rootpath, root_env)

  try:
    ret = find_deref_completions(rootpath, root_env) or enumerate_scope(rootpath, root_env=root_env)
    assert isinstance(ret, dict)
    return ret
  except gcl.EvaluationError:
    # Probably an unbound value or something--just return an empty list
    return {}


def find_inherited_key_completions(rootpath, root_env):
  """Return completion keys from INHERITED tuples.

  Easiest way to get those is to evaluate the tuple, check if it is a CompositeTuple,
  then enumerate the keys that are NOT in the rightmost tuple.
  """
  tup = inflate_context_tuple(rootpath, root_env)
  if isinstance(tup, runtime.CompositeTuple):
    keys = set(k for t in tup.tuples[:-1] for k in t.keys())
    return {n: get_completion(tup, n) for n in keys}
  return {}


def find_value_at_cursor(ast_tree, filename, line, col, root_env=gcl.default_env):
  """Find the value of the object under the cursor."""
  q = gcl.SourceQuery(filename, line, col)
  rootpath = ast_tree.find_tokens(q)
  rootpath = path_until(rootpath, is_thunk)

  if len(rootpath) <= 1:
    # Just the file tuple itself, or some non-thunk element at the top level
    return None

  tup = inflate_context_tuple(rootpath, root_env)
  try:
    if isinstance(rootpath[-1], ast.Inherit):
      # Special case handling of 'Inherit' nodes, show the value that's being
      # inherited.
      return tup[rootpath[-1].name]
    return rootpath[-1].eval(tup.env(tup))
  except gcl.EvaluationError as e:
    return e


def pair_iter(xs):
  last = None
  for x in xs:
    if last is not None:
      yield (last, x)
    last = x


class Completion(collections.namedtuple('Completion', ['name', 'builtin', 'doc', 'span'])):
  """Represents a potential completion."""
  pass
