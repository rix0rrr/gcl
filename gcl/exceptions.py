class GCLError(RuntimeError):
  pass

class ParseError(GCLError):
  pass

class EvaluationError(GCLError):
  def __init__(self, message, inner=None):
    super(EvaluationError, self).__init__(message)
    self.inner = inner

  def __str__(self):
    return self.args[0] + ('\n' + str(self.inner) if self.inner else '')

class RecursionError(EvaluationError):
  pass

class SchemaError(EvaluationError):
  pass
