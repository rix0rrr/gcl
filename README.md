gcl -- Generic Configuration Language
=====================================

GCL is an declarative modeling language that can be dropped into any Python project. It supports
dictionaries with name-value pairs, all the basic types you'd expect, lists, includes, and methods
for abstraction.

The goal for GCL is to be a modeling language with lots of expressive power, intended to make
complex configurations extremely DRY. Behavior is not part of the goal of the language; behavior and
semantics are added by scripts that interpret the GCL model.

Documentation
-------------

Detailed documentation is available at [http://gcl.readthedocs.org](http://gcl.readthedocs.org)

Quick GCL showcase
------------------

GCL is built around named tuples, written with curly braces:

    # This is a comment
    # Various data types:
    number = 1;
    string =  'value';  # Strings can be doubly-quoted as well
    bool =  true;       # Note: lowercase
    expression = number * 2; 
    list = [ 1, 2, 3 ];

Expressions:

    a = 1 + 1;
    b = 'foo' + 'bar';
    c = 80 * '-';

    d = inc(1);  # Function application
    e = inc 1;   # Can omit parens with 1 argument

    # Conditionals
    allow_test_commands = if stage == 'alpha' then true else false;

    # List comprehension
    evens = [ x * 2 for x in [1, 2, 3, 4, 5] if x % 2 == 0 ];

Tuples and accessing tuple members:

    tuple = {
      foo = 3;
    }

    that_foo = tuple.foo;

Includes and tuple composition:

    http = include 'library/http.gcl';
    server = http.Server {
        port = 8080;
    }

Using the library
-----------------

You now know enough GCL to get started. Using the library looks like this:

    import gcl
    from gcl import util

    # Load and evaluate the given file
    model = gcl.load('myfile.gcl')

    # This gives you a dict-like object back, that you can just index
    print(model['element'])

    # Translate the whole thing to a Python dict (for example to convert to JSON)
    dict_model = util.to_python(model)

    import json
    print(json.dumps(dict_model))

Requirements
------------

* Uses `pyparsing`.

Extra
-----

* Vim syntax definitions available: https://github.com/rix0rrr/vim-gcl
