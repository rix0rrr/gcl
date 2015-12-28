"""Schema evaluation classes."""

from . import exceptions
import gcl


class Schema(object):
  def validate(self, value):
    raise NotImplementedError()

  def get_subschema(self, key):
    raise NotImplementedError()

  def __neq__(self, other):
    return not (self == other)


class AnySchema(Schema):
  def validate(self, value):
    pass

  def get_subschema(self, key):
    return self

  def __repr__(self):
    return 'AnySchema()'

  def __eq__(self, other):
    return isinstance(other, AnySchema)


class AndSchema(Schema):
  def __init__(self, one, two):
    self.one = one
    self.two = two

  def validate(self, value):
    self.one.validate(value)
    self.two.validate(value)

  def get_subschema(self, key):
    return AndSchema.make(self.one.get_subschema(key), self.two.get_subschema(key))

  def __repr__(self):
    return 'AndSchema(%r, %r)' % (self.one, self.two)

  def __eq__(self, other):
    return isinstance(other, AndSchema) and self.one == other.one and self.two == other.two

  @staticmethod
  def make(one, two):
    return AndSchema(one, two) if one != two else one


class ScalarSchema(Schema):
  def __init__(self, typename):
    self.typename = typename

  def validate(self, value):
    SCALAR_TYPES[self.typename](self, value)

  def get_subschema(self, key):
    raise RuntimeError('Cannot get subschema from ScalarSchema')

  def _validate_string(self, value):
    if not gcl.is_str(value):
      raise exceptions.SchemaError('%r should be a string' % value)

  def _validate_int(self, value):
    if not isinstance(value, int):
      raise exceptions.SchemaError('%r should be an int' % value)

  def _validate_bool(self, value):
    if not isinstance(value, bool):
      raise exceptions.SchemaError('%r should be a bool' % value)

  def _validate_null(self, value):
    if value is not None:
      raise exceptions.SchemaError('%r should be a null' % value)

  def _validate_float(self, value):
    if not isinstance(value, float):
      raise exceptions.SchemaError('%r should be a float' % value)

  def __repr__(self):
    return 'ScalarSchema(%r)' % (self.typename)

  def __eq__(self, other):
    return isinstance(other, ScalarSchema) and self.typename == other.typename


class ListSchema(Schema):
  def __init__(self, element_schema):
    self.element_schema = element_schema

  def validate(self, value):
    if not gcl.is_list(value):
      raise exceptions.SchemaError('%r should be a list' % value)

    for i, x in enumerate(value):
      try:
        self.element_schema.validate(x)
      except Exception, e:
        raise exceptions.SchemaError('While validating element %d of %r:\n%s' % (i + 1, x, e))

  def get_subschema(self, key):
    raise RuntimeError('Cannot get subschema from ListSchema')

  def __repr__(self):
    return 'ListSchema(%r)' % (self.element_schema)

  def __eq__(self, other):
    return isinstance(other, ListSchema) and self.element_schema == other.element_schema


class TupleSchema(Schema):
  def __init__(self, field_schemas, required_fields):
    self.field_schemas = field_schemas
    self.required_fields = required_fields

  def validate(self, value):
    if not gcl.is_tuple(value):
      raise exceptions.SchemaError('%r should be a tuple' % value)

    key_check = getattr(value, 'is_bound', value.has_key)

    for k in self.required_fields:
      if not key_check(k):
        raise exceptions.SchemaError('Tuple is missing required key %r in %r' % (k, value))

    # Don't validate the subfields here -- we'll do that lazily when
    # we access the subfields.

  def get_subschema(self, key):
    return self.field_schemas.get(key, AnySchema())

  def __repr__(self):
    return 'TupleSchema(%r, %r)' % (self.field_schemas, self.required_fields)

  def __eq__(self, other):
    return (isinstance(other, TupleSchema) and self.required_fields == other.required_fields
        and self.field_schemas == other.field_schemas)


SCALAR_TYPES = {'string': ScalarSchema._validate_string,
                'int': ScalarSchema._validate_int,
                'bool': ScalarSchema._validate_bool,
                'null': ScalarSchema._validate_null,
                'float': ScalarSchema._validate_float}


def from_spec(spec):
  """Return a schema object from a spec.

  A spec is either a string for a scalar type, or a list of 0 or 1 specs,
  or a dictionary with two elements: {'fields': { ... }, required: [...]}.
  """
  if not spec:
    return AnySchema()

  if gcl.is_str(spec):
    # Scalar type
    if spec not in SCALAR_TYPES:
      raise exceptions.SchemaError('Not a valid schema type: %r' % spec)
    return ScalarSchema(spec)

  if gcl.is_list(spec):
    return ListSchema(from_spec(spec[0]) if len(spec) else AnySchema())

  if gcl.is_tuple(spec):
    return TupleSchema(spec.get('fields', {}), spec.get('required', []))

  raise SchemaError('Not valid schema spec; %r' % spec)
