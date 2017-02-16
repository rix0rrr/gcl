from gcl import sparse

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
    ('dq_string', sparse.quoted_string_regex('"')),
    ('sq_string', sparse.quoted_string_regex("'")),
    ('comparison_op', '|'.join(['<=', '>=', '==', '<', '>'])),
    ('mul_op', '|'.join('[*/%]')),
    ('add_op', '|'.join('[+-]')),
    ('floating', r'-?\d+\.\d+'),
    ('integer', r'-?\d+'),
    # Symbols
    ('sym', '[' + ''.join('\\' + s for s in '[](){}=,;:.') + ']'),
]

tokenizer = sparse.Tokenizer(TOKENS)
