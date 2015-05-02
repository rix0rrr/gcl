import unittest
from os import path

import gcl
from gcl import util


class TestJSONLoader(unittest.TestCase):
  """Test our JSON compatibility routines."""

  def set_json(self, json):
    self.loader = util.JSONLoader(gcl.InMemoryFiles({
      '/test.json': json
    }), filter_fn=util.interpolate_json)

  def parse(self, gcl_contents):
    return gcl.loads(gcl_contents, filename='/root.gcl', loader=self.loader)

  def testJSONSimple(self):
    self.set_json('{"test" : "it works"}')

    x = self.parse("""
    included = include "test.json";
    """)

    self.assertEquals('it works', x['included']['test'])

  def testJSONWithSubstitutions(self):
    self.set_json('{"test" : "it {verb}"}')

    x = self.parse("""
    included = include "test.json" {
      verb = 'works'
    }
    """)

    self.assertEquals('it works', x['included']['test'])

  def testJSONList(self):
    self.set_json('["{foo}", "{foo}"]')

    x = self.parse("""
    included = include "test.json" {
      foo = 'boo'
    }
    """)

    self.assertEquals(['boo', 'boo'], list(x['included']))

  def testJSONString(self):
    self.set_json('"{foo}{foo}"')

    x = self.parse("""
    included = include "test.json" {
      foo = 'boo'
    }
    """)

    self.assertEquals('booboo', str(x['included']))

  def testJSONNumber(self):
    self.set_json('3')

    x = self.parse("""
    included = include "test.json";
    """)

    self.assertEquals(3, x['included'])
