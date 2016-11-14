class GCLError(RuntimeError):
  pass


class ParseError(GCLError):
  def __init__(self, filename, sourcelocation, error_message):
    self.filename = filename
    self.sourcelocation = sourcelocation
    self.error_message = error_message

    nice_message = '%s:%d: %s\n%s\n%s^-- here' % (filename, sourcelocation.lineno, error_message, sourcelocation.line, ' ' * (sourcelocation.col - 1))
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
