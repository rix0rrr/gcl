from . import sparse

class GCLError(RuntimeError):
  pass


class ParseError(GCLError):
  def __init__(self, span, error_message):
    assert isinstance(span, sparse.Span)
    self.span = span
    self.error_message = error_message

    nice_message = span.annotated_source(error_message)
    super(ParseError, self).__init__(nice_message)


class EvaluationError(GCLError):
  def __init__(self, message, inner=None):
    super(EvaluationError, self).__init__(message)
    self.inner = inner

  def __str__(self):
    return self.args[0] + ('\n' + str(self.inner) if self.inner else '')


class UnboundNameError(EvaluationError):
  pass


class RecursionError(EvaluationError):
  pass


class SchemaError(EvaluationError):
  pass
