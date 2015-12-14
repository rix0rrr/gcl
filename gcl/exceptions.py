class GCLError(RuntimeError):
  pass

class ParseError(GCLError):
  pass

class SchemaError(GCLError):
  pass
