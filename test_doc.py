import unittest

from gcl import doc


class TestDocGenerator(unittest.TestCase):
  def test_codeblock_recognizer(self):
    self.assertEquals(['bliep', '', '.. code-block:: javascript', '', '    bloep'],
        list(doc.stylize_comment_block(['bliep', '    bloep'])))

    self.assertEquals(['bliep', '', '.. code-block:: javascript', '', '    bloep'],
        list(doc.stylize_comment_block(['bliep', '', '    bloep'])))

    self.assertEquals(['bliep', '', '.. code-block:: javascript', '', '    bloep', 'blap'],
        list(doc.stylize_comment_block(['bliep', '', '    bloep', 'blap'])))

    self.assertEquals(['header', '', 'and two', 'lines'],
        list(doc.stylize_comment_block(['header', '', 'and two', 'lines'])))
