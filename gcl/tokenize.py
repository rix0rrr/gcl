"""Tokenizer for GCL language.

Based on regex module tricks from:

    http://lucumr.pocoo.org/2015/11/18/pythons-hidden-re-gems/

To get the highest possible speed.
"""
import re

from sre_parse import Pattern, SubPattern, parse
from sre_compile import compile as sre_compile
from sre_constants import BRANCH, SUBPATTERN


class Scanner(object):
    def __init__(self, rules, flags=re.M):
        pattern = Pattern()
        pattern.flags = flags
        pattern.groups = len(rules) + 1

        self.rules = rules
        self._scanner = sre_compile(SubPattern(pattern, [
            (BRANCH, (None, [SubPattern(pattern, [
                (SUBPATTERN, (group, parse(regex, flags, pattern))),
            ]) for group, (_, regex) in enumerate(rules, 1)]))
        ])).scanner

    def scan(self, string, skip=False):
        sc = self._scanner(string)

        match = None
        for match in iter(sc.search if skip else sc.match, None):
            yield self.rules[match.lastindex - 1][0], match

        if match.end() < len(string):
            raise RuntimeError('No more match at %d (%s)' % (match.end(), string[match.end():match.end() + 10] + '...'))


def quoted_string_regex(quote):
    return quote + '(?:[^' + quote + r'\\]|\\.)*' + quote

# Scanning for all of these at once is faster than scanning for them individually.
keywords = ['inherit', 'if', 'then', 'else', 'include', 'null', 'true', 'false', 'for', 'in']

TOKENS = [
    ('whitespace', r'\s+'),
    ('comment', '#$|#[^.].*$'),
    ('doc_comment', '#\.(.*)$'),
    # Keywords
    ('logical_combinator', 'and|or'),
    ('not', 'not'),
    ('keyword', '|'.join(k for k in keywords)),
    # Identifiers
    ('quoted_identifier', quoted_string_regex('`')),
    ('identifier', r'[a-zA-Z_]([a-zA-Z0-9_:-]*[a-zA-Z0-9_])?'),
    ('dq_string', quoted_string_regex('"')),
    ('sq_string', quoted_string_regex("'")),
    ('comparison_op', '|'.join(['<=', '>=', '==', '<', '>'])),
    ('mul_op', '|'.join('[*/%]')),
    ('add_op', '|'.join('[+-]')),
    ('floating', r'-?\d+\.\d+'),
    ('integer', r'-?\d+'),
    # Symbols
    ('sym', '[' + ''.join('\\' + s for s in '[](){}=,;:.') + ']'),
]

scanner = Scanner(TOKENS)


class Token(object):
    __slots__ = ('type', 'value', 'span')

    def __init__(self, type, value, span):
        self.type = type
        self.value = value
        self.span = span


def tokenize(string):
    for token, match in scanner.scan(string):
        if token == 'whitespace' or token == 'comment':
            continue
        if token == 'keyword' or token == 'sym':
            token = match.group()
        yield Token(token, match.group(), match.span())
