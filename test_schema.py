import unittest
from os import path

import gcl
from gcl import schema
from gcl import exceptions
from gcl import util


class TestSchemaObjects(unittest.TestCase):
  """Tests to ensure that the schema objects do the right thing."""

  def testScalarString(self):
    schema.from_spec('string').validate('a')

  def testScalarInt(self):
    schema.from_spec('int').validate(3)

  def testScalarIntFails(self):
    self.assertRaises(lambda: schema.from_spec('int').validate('a'))

  def testScalarFloat(self):
    schema.from_spec('float').validate(3.0)

  def testScalarBool(self):
    schema.from_spec('bool').validate(False)

  def testListWithSpec(self):
    int_schema = schema.ScalarSchema('int')
    schema.from_spec([int_schema]).validate([3])
    schema.from_spec([int_schema]).validate([])
    schema.from_spec([int_schema]).validate([1, 2])
    schema.from_spec([]).validate([1, 2])

  def testListSchema(self):
    with self.assertRaises(exceptions.SchemaError):
      schema.from_spec([]).validate('whoops')

  def testListWithSpecFails(self):
    string_schema = schema.ScalarSchema('string')
    with self.assertRaises(exceptions.SchemaError):
      schema.from_spec([string_schema]).validate([1, 2])

  def testTuple(self):
    schema.from_spec({}).validate({})

  def testTupleWithSubSchemas(self):
    s = schema.from_spec({'fields': {'a': schema.from_spec('string')}})
    s.validate({'a': 'b'})
    s.validate({})
    # This doesn't fail! TupleSchema only checks required fields!
    s.validate({'a': 3})

  def testTupleSchemaCorrectSubSchemaType(self):
    """To make sure we can't use tuple schemas incorrectly."""
    with self.assertRaises(ValueError):
      schema.from_spec({'fields': {'a': 'string'}})

  def testTupleSchemaSubSchema(self):
    s = schema.from_spec({'fields': {'a': schema.from_spec('string')}})

    # Subschema for a requires a string
    with self.assertRaises(exceptions.SchemaError):
      s.get_subschema('a').validate(3)

    # No schema, so don't care
    s.get_subschema('b').validate(3)

  def testTupleWithDeepSchemas(self):
    s = schema.from_spec({'fields': {'a': schema.from_spec({'fields': { 'b': schema.from_spec('int') }})}})
    s.validate({'a': {}})
    # This doesn't fail! No deep verification!
    s.validate({'a': {'b': 'c'}})

  def testTupleWithRequiredField(self):
    s = schema.from_spec({'required': ['a']})
    s.validate({'a': 3})
    with self.assertRaises(exceptions.SchemaError):
      s.validate({})


class TestSchemaEquality(unittest.TestCase):
  def setUp(self):
    self.int1 = schema.ScalarSchema('int')
    self.int2 = schema.ScalarSchema('int')
    self.string = schema.ScalarSchema('string')

  def testScalarEquality(self):
    self.assertTrue(self.int1 == self.int2)
    self.assertFalse(self.int1 != self.int2)
    self.assertFalse(self.int1 == self.string)
    self.assertTrue(self.int1 != self.string)


class TestCollapseAndSchema(unittest.TestCase):

  def collapsesTo(self, one, two, expected):
    s = schema.AndSchema.make(one, two)
    self.assertEquals(expected, s)

  def testCollapseAniesLeft(self):
    self.collapsesTo(schema.AnySchema(), schema.ScalarSchema('int'),
                     schema.ScalarSchema('int'))

  def testCollapseAniesRight(self):
    self.collapsesTo(schema.ScalarSchema('int'), schema.AnySchema(),
                     schema.ScalarSchema('int'))

  def testCollapseTwoAnies(self):
    self.collapsesTo(schema.AnySchema(), schema.AnySchema(),
                     schema.AnySchema())

  def testCollapseEqualSchemas(self):
    self.collapsesTo(schema.ScalarSchema('int'), schema.ScalarSchema('int'),
                     schema.ScalarSchema('int'))


class TestSchemaFromGCL(unittest.TestCase):
  """Tests to ensure that schema objects parsed directly from JSON look good."""

  def testSubSchema(self):
    x = gcl.loads("""
      a : int;
    """)

    self.assertEquals(schema.ScalarSchema('int'), x.tuple_schema.get_subschema('a'))

  def testTupleRequiredFields(self):
    x = gcl.loads("""
      a : required int = 3;
      b : string;
      c : required = 1;
    """)

    self.assertEquals(['a', 'c'], sorted(x.tuple_schema.required_fields))

  def testSchemaRefersToVariable(self):
    x = gcl.loads("""
      Xint = { x : int; };
      y : Xint = { x = 'bla' };
    """)
    self.assertEquals(schema.ScalarSchema('int'), x['y'].tuple_schema.get_subschema('x'))

  def testSchemasCompose(self):
    x = gcl.loads("""
      Xint = { x : int; };
      y = Xint { x = 'bla'; y : string };
    """)
    self.assertEquals(schema.ScalarSchema('string'), x['y'].tuple_schema.get_subschema('y'))
    self.assertEquals(schema.ScalarSchema('int'), x['y'].tuple_schema.get_subschema('x'))

  def testSchemasCompose3Times(self):
    x = gcl.loads("""
      y = { x : int } { y : string } { z : bool };
    """)
    self.assertEquals(schema.ScalarSchema('string'), x['y'].tuple_schema.get_subschema('y'))
    self.assertEquals(schema.ScalarSchema('int'), x['y'].tuple_schema.get_subschema('x'))
    self.assertEquals(schema.ScalarSchema('bool'), x['y'].tuple_schema.get_subschema('z'))

  def testInheritSchemaWithOverriddenValue(self):
    x = gcl.loads("""
      y = { x : int } { x = 3 };
    """)
    self.assertEquals(schema.ScalarSchema('int'), x['y'].tuple_schema.get_subschema('x'))

  def testListSchema(self):
    x = gcl.loads("""
      y = { x : [int] }
    """)
    self.assertEquals(schema.ListSchema(schema.ScalarSchema('int')), x['y'].tuple_schema.get_subschema('x'))

  def testAnyListSchema(self):
    x = gcl.loads("""
      y = { x : [] }
    """)
    self.assertEquals(schema.ListSchema(schema.AnySchema()), x['y'].tuple_schema.get_subschema('x'))

  def testInheritSchemaWithOverriddenValueWhichIsAList(self):
    x = gcl.loads("""
      y = { x : [int] } { x = [3] };
    """)
    self.assertEquals(schema.ListSchema(schema.ScalarSchema('int')), x['y'].tuple_schema.get_subschema('x'))


class TestSchemaInGCL(unittest.TestCase):
  """Tests to ensure that using schemas directly inside GCL do the right thing."""

  def testFailingToParseTupleWithSchema(self):
    with self.assertRaises(exceptions.SchemaError):
      gcl.loads("""
        a : required;
      """)

  def testSchemaValidationIsLazy(self):
    x = gcl.loads("""
      a : int = 'foo';
    """)
    with self.assertRaises(exceptions.SchemaError):
      print(x['a'])

  def testListSchema(self):
    x = gcl.loads("""
      a : [int] = ['boo'];
    """)
    with self.assertRaises(exceptions.SchemaError):
      print(x['a'])

  def testSchemaSurvivesComposition(self):
    x = gcl.loads("""
        A = { foo : int };
        B = A { foo = 'hello' };
        """)
    with self.assertRaises(exceptions.SchemaError):
      print(x['B']['foo'])

  def testRequiredFieldsInComposition(self):
    x = gcl.loads("""
      SuperClass = { x : required };
      ok_instance = SuperClass { x = 1 };
      failing_instance = SuperClass { y = 1 };
    """)
    print(x['ok_instance'])
    with self.assertRaises(exceptions.SchemaError):
      print(x['failing_instance'])

  def testClassesInTupleComposition(self):
    x = gcl.loads("""
      Atom = {
        name : required;
      };

      Molecule = {
        atoms : [Atom];
      };

      objects = {
        correct = Molecule {
          atoms = [{ name = 'O' }, { name = 'C' }];
        };
        broken = Molecule {
          atoms = [{ valence = 3 }];
        };
      }
    """)
    zzz = x['objects']['correct']['atoms']
    with self.assertRaises(exceptions.SchemaError):
      zzz = x['objects']['broken']['atoms']

  def testSchemaTypoExceptionIsDescriptive(self):
    """Check that if you make a typo in a type name the error is helpful."""

    with self.assertRaises(exceptions.EvaluationError):
      x = gcl.loads("""
      a : strong;
      """)

  def testSpecifySchemaDeep(self):
    obj = gcl.loads("x : { a : required } = { b = 'foo' };")
    with self.assertRaises(exceptions.SchemaError):
      print(obj['x'])

  def testSpecifySchemaDeepInList(self):
    obj = gcl.loads("x : [{ a : int }] = [{ a = 'foo' }];")
    with self.assertRaises(exceptions.SchemaError):
      print(obj['x'][0]['a'])

  def testSchemaCanBeSetFromAbove(self):
    obj = gcl.loads("x : { x : { a : int }} = { x = { a = 'hoi' }}")
    with self.assertRaises(exceptions.SchemaError):
      print(obj['x']['x']['a'])

  def testSchemaCombinesFromAbove(self):
    obj = gcl.loads("x : { x : { a : int }} = { x = { a = 3; b : required }}")
    with self.assertRaises(exceptions.SchemaError):
      print(obj['x']['x']['a'])

  def testDontNeedToWriteSpace(self):
    """It's annoying me to have to write a space for schema definitions."""
    obj = gcl.loads('x: int = 3;')
    print(obj['x'])


class TestExportVisibilityThroughSchemas(unittest.TestCase):
  """Test the annotation of schemas with non-exportable fields for JSON exports."""

  def testNoExportFieldFromTuple(self):
    obj = gcl.loads("x : private = 3; y = 5")
    x = util.to_python(obj)
    self.assertEquals({'y': 5}, x)

  def testNoExportFieldFromCompositeTuple(self):
    obj = gcl.loads("x = { x : private = 3 } { y = 5 }")
    x = util.to_python(obj['x'])
    self.assertEquals({'y': 5}, x)

  def testNoExportFieldFromCompositeTupleWithPrivateOverride(self):
    obj = gcl.loads("x = { x = 3; y = 5 } { x : private }")
    x = util.to_python(obj['x'])
    self.assertEquals({'y': 5}, x)

  def testNoExportFieldFromTupleComposedWithDict(self):
    obj = gcl.loads("x = { x : private = 3 } Y", env={'Y': {'y': 5}})
    x = util.to_python(obj['x'])
    self.assertEquals({'y': 5}, x)
