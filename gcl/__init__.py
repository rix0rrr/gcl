"""
GCL -- Generic Configuration Language

See README.md for an explanation of GCL and concepts.
"""
from os import path
import json

from . import functions
from . import exceptions
from . import schema
from . import ast
from . import runtime
from . import util
from . import framework


__version__ = '0.6.10'


# Namespace copy for backwards compatibility.
GCLError = exceptions.GCLError
ParseError = exceptions.ParseError
EvaluationError = exceptions.EvaluationError

is_tuple = framework.is_tuple
is_list = framework.is_list

TupleLike = framework.TupleLike
Environment = framework.Environment
find_relative = util.find_relative
Tuple = runtime.Tuple

InMemoryFiles = runtime.InMemoryFiles


class NormalLoader(object):
  def __init__(self, fs):
    self.fs = fs
    self.cache = framework.Cache()

  def __call__(self, current_file, rel_path, env=None):
    nice_path, full_path = self.fs.resolve(current_file, rel_path)

    # Cache on full path, but tell script about nice path
    if path.splitext(nice_path)[1] == '.json':
      # Load as JSON
      do_load = lambda: json.loads(self.fs.load(full_path))
    else:
      # Load as GCL
      do_load = lambda: loads(self.fs.load(full_path), filename=nice_path, loader=self, env=env)
    return self.cache.get(full_path, do_load)


def loader_with_search_path(search_path):
  return NormalLoader(runtime.OnDiskFiles(search_path))


make_env = framework.make_env
make_tuple = ast.make_tuple
SourceQuery = ast.SourceQuery


# Default loader doesn't have any search path
default_loader = NormalLoader(runtime.OnDiskFiles())

#----------------------------------------------------------------------
#  Top-level functions
#

_default_bindings = {}
_default_bindings.update(functions.builtin_functions)
_default_bindings.update(schema.builtin_schemas)

default_env = framework.Environment(_default_bindings)

def reads(s, filename=None, loader=None, implicit_tuple=True, allow_errors=False):
  """Load but don't evaluate a GCL expression from a string."""
  return ast.reads(s,
      filename=filename or '<input>',
      loader=loader or default_loader,
      implicit_tuple=implicit_tuple,
      allow_errors=allow_errors)


def read(filename, loader=None, implicit_tuple=True, allow_errors=False):
  """Load but don't evaluate a GCL expression from a file."""
  with open(filename, 'r') as f:
    return reads(f.read(),
                 filename=filename,
                 loader=loader,
                 implicit_tuple=implicit_tuple,
                 allow_errors=allow_errors)


mod_schema = schema  # Just because I want to use schema as a keyword argument below
def loads(s, filename=None, loader=None, implicit_tuple=True, env={}, schema=None):
  """Load and evaluate a GCL expression from a string."""
  ast = reads(s, filename=filename, loader=loader, implicit_tuple=implicit_tuple)
  if not isinstance(env, framework.Environment):
    # For backwards compatibility we accept an Environment object. Otherwise assume it's a dict
    # whose bindings will add/overwrite the default bindings.
    env = framework.Environment(dict(_default_bindings, **env))
  obj = framework.eval(ast, env)
  return mod_schema.validate(obj, schema)


def load(filename, loader=None, implicit_tuple=True, env={}, schema=None):
  """Load and evaluate a GCL expression from a file."""
  with open(filename, 'r') as f:
    return loads(f.read(),
                 filename=filename,
                 loader=loader,
                 implicit_tuple=implicit_tuple,
                 env=env,
                 schema=schema)
