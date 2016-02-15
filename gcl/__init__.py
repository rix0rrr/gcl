"""
GCL -- Generic Configuration Language

See README.md for an explanation of GCL and concepts.
"""
from . import functions
from . import exceptions
from . import schema
from . import ast
from . import runtime
from . import util
from . import framework

__version__ = '0.5.3'


# Namespace copy for backwards compatibility
GCLError = exceptions.GCLError
ParseError = exceptions.ParseError

is_tuple = framework.is_tuple
is_list = framework.is_list

TupleLike = framework.TupleLike
Environment = framework.Environment
find_relative = util.find_relative
Tuple = runtime.Tuple

InMemoryFiles = runtime.InMemoryFiles


def loader_with_search_path(search_path):
  return runtime.NormalLoader(runtime.OnDiskFiles(search_path))


# Default loader doesn't have any search path
default_loader = runtime.NormalLoader(runtime.OnDiskFiles())

#----------------------------------------------------------------------
#  Top-level functions
#

_default_bindings = {}
_default_bindings.update(functions.builtin_functions)
_default_bindings.update(schema.builtin_schemas)

default_env = framework.Environment(_default_bindings)

def reads(s, filename=None, loader=None, implicit_tuple=True):
  """Load but don't evaluate a GCL expression from a string."""
  return ast.reads(s,
      filename or '<input>',
      loader or default_loader,
      implicit_tuple)


def read(filename, loader=None, implicit_tuple=True):
  """Load but don't evaluate a GCL expression from a file."""
  with open(filename, 'r') as f:
    return reads(f.read(),
                 filename=filename,
                 loader=loader,
                 implicit_tuple=implicit_tuple)


def loads(s, filename=None, loader=None, implicit_tuple=True, env=None):
  """Load and evaluate a GCL expression from a string."""
  ast = reads(s, filename=filename, loader=loader, implicit_tuple=implicit_tuple)
  return framework.eval(ast, env or default_env)


def load(filename, loader=None, implicit_tuple=True, env=None):
  """Load and evaluate a GCL expression from a file."""
  with open(filename, 'r') as f:
    return loads(f.read(),
                 filename=filename,
                 loader=loader,
                 implicit_tuple=implicit_tuple,
                 env=env)
