import unittest
from os import path

import gcl
from gcl import schema
from gcl import exceptions


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
    schema.from_spec(['int']).validate([3])
    schema.from_spec(['int']).validate([])
    schema.from_spec(['int']).validate([1, 2])
    schema.from_spec([]).validate([1, 2])

  def testListWithSpecFails(self):
    with self.assertRaises(exceptions.SchemaError):
      schema.from_spec(['string']).validate([1, 2])

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


class TestSchemaInGCL(unittest.TestCase):
  """Tests to ensure that using schemas directly inside GCL do the right thing."""

  def testFailingToParseTupleWithSchema(self):
    with self.assertRaises(exceptions.SchemaError):
      gcl.loads("""
        a : required;
      """)

  def testSchemaValidationIsLazy(self):
    #print gcl.reads("a : int = 'foo';")
    x = gcl.loads("""
      a : int = 'foo';
    """)
    with self.assertRaises(exceptions.SchemaError):
      print x['a']

  def testListSchema(self):
    x = gcl.loads("""
      a : [int] = ['boo'];
    """)
    with self.assertRaises(exceptions.SchemaError):
      print x['a']

  def testSchemaSurvivesComposition(self):
    x = gcl.loads("""
        A = { foo : int };
        B = A { foo = 'hello' };
        """)
    with self.assertRaises(exceptions.SchemaError):
      print x['B']

  def testCompositionCombinesSchemas(self):
    pass

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
    print x['objects']['correct']
    with self.assertRaises(exceptions.SchemaError):
      print x['objects']['broken']

  def testSchemaTypoExceptionIsDescriptive(self):
    """Check that if you make a typo in a type name the error is helpful."""

    with self.assertRaises(exceptions.EvaluationError):
      x = gcl.loads("""
      a : strong;
      """)
      self.fail('Should have thrown')

  def testSpecifySchemaDeep(self):
    with self.assertRaises(exceptions.SchemaError):
      gcl.loads("x : { a : required } = { b = 'foo' };")

  def testSpecifySchemaDeepInList(self):
    with self.assertRaises(exceptions.SchemaError):
      gcl.loads("x : [{ a : int }] = [{ a = 'foo' }];")

  def testSchemaCanBeSetFromAbove(self):
    pass

  def testSchemaCombinesFromAbove(self):
    pass

