Language Basics
===============

Basic syntax
------------

GCL is built around named tuples, written with curly braces:

    {
      # This is a comment
      number = 1;
      string =  'value';  # Strings can be doubly-quoted as well
      bool =  true;       # Note: lowercase
      expression = number * 2; 
      list = [ 1, 2, 3 ];
    }

The top-level of a file will be parsed as a tuple automatically, so you don't
write the braces there. Semicolons are considered separators. They may be
ommitted after the last statement if that aids readability.

The basic types you'd expect are supported: strings, ints, floats, bools and
mapped onto their Python equivalents. Lists are supported, but can't really be
manipulated in GCL right now.

Expressions
-----------

    a = 1 + 1;
    b = 'foo' + 'bar';
    c = 80 * '-';

GCL has an expression language, looking much like other languages you're used
to. The evaluation model is mostly borrowed from Python, so things you expect
from Python (such as being able to use `+` for both addition and string
concatenation).

Calling functions
-----------------

    inc(1)

Function application also looks the same as in Python. There's currently no way
to define functions in GCL, but you can invoke functions passed in from the
external environment.

    inc 1

If a function only has one argument, you can omit the parentheses and simply
put a space between the function and the argument.

Accessing values in tuples
--------------------------

    tuple = {
      foo = 3;
    };

    that_foo = tuple.foo;

Periods are used to dereference tuples using constant keys. If the key is in a
variable, tuples can be treated as maps (functions) to get a single key out:

    tuple = {
      foo = 3;
    }

    that_foo1 = tuple('foo');

    which_key = 'foo';
    that_foo2 = tuple(which_key);

Including other files
---------------------

    http = include 'library/http.gcl';
    server = http.Server {
        port = 8080;
    }

External files can be included with the built-in `include()` function. The
result of that expression is the result of parsing that file (which will be
parsed as a tuple using the default environment). Relative filenames are
resolved with respect to the _including_ file.

'if' expressions
-----------------

Expressions can also include conditions, using the `if` statement:

    allow_test_commands = if stage == 'alpha' then true else false;
    
    # Of course, since these are booleans, the above could also be written as:
    allow_test_commands = stage == 'alpha';

List comprehensions
--------------------

Lists can be manipulated using list comprehensions:

    [ x * 2 for x in [1, 2, 3, 4, 5] if x % 2 == 0 ]

Tuple composition
-----------------

As a special case, a tuple can be applied to another tuple, yielding a new
tuple thats the merge of both (with the right tuple overwriting existing keys
in the left one).

This looks especially convenient when A is a reference and B is a tuple
literal, and you use the paren-less function invocation:

    FooApp = {
      program = 'foo';
      cwd = '/tmp';
    }

    my_foo = FooApp {
      cwd = '/home';
    }

`my_foo` is now a tuple with 2 fields, `program = 'foo'` (unchanged) and
`cwd = '/home'` (overwritten).

This makes it possible to do abstraction: just define tuples with the common
components and inherit specializations from them.

Parameterized tuples
--------------------

Because tuple elements are lazily evaluated (i.e., only when requested), you
can also use this for parameterization. Declare keys without giving them a
value, to signal that inheriting tuples should fill these values:

    greet = {
      greeting;
      message = greeting + ' world';
    };

    hello_world = greet { greeting = 'hello' }

If `message` is evaluated, but `greeting` happens to not be filled in, an
error will be thrown. To force eager evaluation (to try and catch typos), use
`eager()` on a tuple.

Accessing inherited values
--------------------------

Normally in a tuple composition, variables that you set are completely replaced
with the new value you're setting. Sometimes you don't want this; you may want
to take an existing object or list and add some values to it. In that case, you
can refer to the "original" value (values to the left of the current tuple
inside the composition) by referring to a tuple called `base.`. For example:

    parent = {
        attributes = {
            food = 'fast';
            speed = 'slow';
        }
    };
    final = parent {
        attributes = base.attributes {
            speed = 'fast';
        }
    };
