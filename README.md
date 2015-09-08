gcl -- Generic Configuration Language
=====================================

GCL is an abstract configuration language that can be dropped into any Python
project. It supports dictionaries with name-value pairs, all the basic types
you'd expect, lists, includes, and methods for abstraction.

Vision
------

The goal for GCL is to be a declarative configuration/modeling language, with
lots of expressive power, intended to make configs DRY. Behavior is not part of
the goal of the language; behavior and semantics are added by scripts that
interpret the GCL model.

Scoping rules have been designed to make single files easier to analyze and to
predict their behavior (for example, scoping is not dynamic but all external
imports have been declared). This analyzability is good for larger projects,
but this necesitates some syntactical overhead that may not make the project
worthwhile for small configs split over multiple files. Alternatively, don't
split up into multiple files :).

Why not use JSON?
-----------------

JSON is good for writing complex data structures in a human-readable way, but
it breaks down when your config starts to become more complex. In particular,
JSON lacks the following:

* No comments, making it hard to describe what's going on.
* No expressions, so there are no ways to have values depend on each other
  (e.g., `instances_to_start = expected_tps / 1000`.
* No abstraction, which makes it impossible to factor out common pieces of
  config.
* All the double quotes I have to type make my pinkies sore! :(

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

    1 + 1
    'foo' + 'bar'
    80 * '-'

GCL has an expression language, looking much like other languages you're used
to. The evaluation model is mostly borrowed from Python, so things you expect
from Python (such as being able to use `+` for both addition and string
concatenation).

    inc(1)

Function application also looks the same as in Python. There's currently no way
to define functions in GCL, but you can invoke functions passed in from the
external environment.

    inc 1

If a function only has one argument, you can omit the parentheses and simply
put a space between the function and the argument.

    tuple = {
      foo = 3;
    }

    that_foo = tuple.foo;

Periods are used to dereference tuples.

    http = include 'library/http.gcl';
    server = http.Server {
        port = 8080;
    }

External files can be included with the built-in `include()` function. The
result of that expression is the result of parsing that file (which will be
parsed as a tuple using the default environment). Relative filenames are
resolved with respect to the _including_ file.

Expressions can also include conditions, using the `if` statement:

    allow_test_commands = if stage == 'alpha' then true else false;
    
    # Of course, since these are booleans, the above could also be written as:
    allow_test_commands = stage == 'alpha';

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

Scoping
-------

References in GCL are lexically scoped, where each tuple forms its own
subscope. This means that something like this:

    x = 3;
    y = {
        x = x;
    };

Wouldn't work, because `x = x` would lead to infinite recursion. Instead,
to bring variables from the outer scope into the inner scope (on the current
tuple), use the `inherit` keyword:

    x = 3;
    y = {
        inherit x;
    }

This will be especially useful if you're going to compose with tuples from a
different file.

In a tuple composition, it's possible to refer to variables in the left side of
the composition using `base.`. This is useful if you want to modify subtuples
(instead of overwriting them, as would be the default):

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

Competition
-----------

* JSON: already mentioned above. Not so nice to write, and because of lack of
  expressive power encourages copy/paste jobs all over the place.
* [TOML](https://github.com/toml-lang/toml): simple and obvious. Doesn't seem
  to allow abstraction and reuse though.
* [UCL](https://github.com/vstakhov/libucl): looks and feels a lot like GCL,
  but the difference with GCL is that in typing `section { }`, in UCL the
  _interpreter_ gives meaning to the identifier `section`, while in GCL the
  model itself gives meaning to `section`. Also, the macro language doesn't
  look so nice to me.
* [Nix language](http://nixos.org/nix/manual/): subconsciously, GCL has been
  modeled a lot after Nix, with its laziness and syntax. Nix' purpose is
  similar (declaring a potentially huge model that's lazily evaluated), though
  its application area is different. Nix uses explicit argument declaration and
  makes tuples nonrecursive, whereas in GCL everything in scope can be
  referenced.

Requirements
------------

* Uses `pyparsing`.

Extra
-----

* Vim syntax definitions available: https://github.com/rix0rrr/vim-gcl
