from . import sparse

class GCLError(RuntimeError):
  pass


class ParseError(GCLError):
  def __init__(self, span, error_message):
    assert isinstance(span, sparse.Span)
    self.span = span
    self.error_message = error_message

    nice_message = '\n'.join(span.annotated_source(error_message))
    super(ParseError, self).__init__(nice_message)

  @property
  def line_nr(self):
    line_nr, _, _ = self.span.line_context()
    return line_nr


class EvaluationError(GCLError):
  def __init__(self, message, span=None, inner=None):
    if span:
      nice_message = '\n'.join(span.annotated_source(message))
    else:
      nice_message = message

    super(EvaluationError, self).__init__(nice_message)
    self.span = span
    self.inner = inner

  def __str__(self):
    return self.message + ('\n' + str(self.inner) if self.inner else '')


class UnboundNameError(EvaluationError):
  pass


class RecursionError(EvaluationError):
  pass


class SchemaError(EvaluationError):
  pass
