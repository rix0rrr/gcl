import unittest
from os import path

import gcl
from gcl import schema
from gcl import exceptions


class TestSchema(unittest.TestCase):
  """Simple tests for Schema classes."""

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
    self.assertRaises(lambda: schema.from_spec(['string']).validate([1, 2]))

  def testTuple(self):
    schema.from_spec({}).validate({})

  def testTupleWithSubSchemas(self):
    s = schema.from_spec({'fields': {'a': 'string'}})
    s.validate({'a': 'b'})
    s.validate({})
    self.assertRaises(lambda: s.validate({'a': 3}))

  def testTupleWithDeepSchemas(self):
    s = schema.from_spec({'fields': {'a': {'fields': { 'b': 'int' }}}})
    s.validate({'a': {}})
    s.validate({'a': {'b': 'c'}})  # No deep verification!

  def testTupleWithRequiredField(self):
    s = schema.from_spec({'required': ['a']})
    s.validate({'a': 3})
    self.assertRaises(lambda: s.validate({}))


class TestGCLAndSchema(unittest.TestCase):
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
    try:
      x = gcl.loads("""
      a : strong;
      """)
      self.fail('Should have thrown')
    except exceptions.SchemaError, e:
      print str(e)
      self.fail('FIXME: Define what a useful error message means')

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

