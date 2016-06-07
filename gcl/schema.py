"""Schema runtime classes."""

from . import exceptions
from . import framework


class Schema(object):
  def validate(self, value):
    raise NotImplementedError()

  def get_subschema(self, key):
    raise NotImplementedError()

  def __ne__(self, other):
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


any_schema = AnySchema()  # Object reuse through immutable singleton


class AndSchema(Schema):
  def __init__(self, one, two):
    assert isinstance(one, Schema)
    assert isinstance(two, Schema)
    self.one = one
    self.two = two

  def validate(self, value):
    self.one.validate(value)
    self.two.validate(value)

  def get_subschema(self, key):
    one_schema = self.one.get_subschema(key)
    two_schema = self.two.get_subschema(key)
    return AndSchema.make(one_schema, two_schema)

  def __repr__(self):
    return 'AndSchema(%r, %r)' % (self.one, self.two)

  def __eq__(self, other):
    return isinstance(other, AndSchema) and self.one == other.one and self.two == other.two

  @staticmethod
  def make(one, two):
    if one == any_schema: return two
    if two == any_schema: return one
    return AndSchema(one, two) if one != two else one


class ScalarSchema(Schema):
  def __init__(self, typename):
    self.typename = typename

  def validate(self, value):
    SCALAR_TYPES[self.typename](self, value)

  def get_subschema(self, key):
    raise RuntimeError('Cannot get subschema from ScalarSchema')

  def _validate_string(self, value):
    if not framework.is_str(value):
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

    if not isinstance(element_schema, Schema):
      raise ValueError('Expecting Schema instance in List, got %r' % element_schema)

  def validate(self, value):
    if not framework.is_list(value):
      raise exceptions.SchemaError('%r should be a list' % value)

    for i, x in enumerate(value):
      try:
        self.element_schema.validate(x)
      except Exception as e:
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

    # I would have loved to do parameter validation here, but that means we can't do recursive tuple
    # schemas (i.e., X = { x : X }). So unfortunately we'll have to defer parameter validation until
    # access. Anyway, this was only used to catch programming errors :).

    # self._validate_params()

  def _validate_params(self):
    for v in self.field_schemas.values():
      if not isinstance(v, Schema):
        raise ValueError('subschemas should be instances of Schema, got %r' % v)

  def validate(self, value):
    self._validate_params()

    if not framework.is_tuple(value):
      raise exceptions.SchemaError('%r should be a tuple' % value)

    key_check = getattr(value, 'is_bound', value.__contains__)

    for k in self.required_fields:
      if not key_check(k):
        raise exceptions.SchemaError('Tuple is missing required key %r in %r' % (k, value))

    # Don't validate the subfields here -- we'll do that lazily when
    # we access the subfields.

  def get_subschema(self, key):
    # The values in field_schemas are unevaluated specs
    return self.field_schemas.get(key, any_schema)

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
  if spec == '':
    return any_schema

  if framework.is_str(spec):
    # Scalar type
    if spec not in SCALAR_TYPES:
      raise exceptions.SchemaError('Not a valid schema type: %r' % spec)
    return ScalarSchema(spec)

  if framework.is_list(spec):
    return ListSchema(spec[0] if len(spec) else any_schema)

  if framework.is_tuple(spec):
    return TupleSchema(spec.get('fields', {}), spec.get('required', []))

  raise exceptions.SchemaError('Not valid schema spec; %r' % spec)


builtin_schemas = {k: ScalarSchema(k) for k in SCALAR_TYPES.keys()}


def nop(x): pass


def validate(obj, schema):
  """Validate an object according to its own AND an externally imposed schema."""
  # Validate returned object according to its own schema
  if hasattr(obj, 'tuple_schema'):
    obj.tuple_schema.validate(obj)
  # Validate object according to externally imposed schema
  if schema:
    schema.validate(obj)
  return obj


def attach(obj, schema):
  """Attach the given schema to the given object."""

  # We have a silly exception for lists, since they have no 'attach_schema'
  # method, and I don't feel like making a subclass for List just to add it.
  # So, we recursively search the list for tuples and attach the schema in
  # there.
  if framework.is_list(obj) and isinstance(schema, ListSchema):
    for x in obj:
      attach(x, schema.element_schema)
    return

  # Otherwise, the object should be able to handle its own schema attachment.
  getattr(obj, 'attach_schema', nop)(schema)
