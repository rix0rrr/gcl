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

The following sections are about advanced concepts.

Scoping
-------

References in GCL are lexically scoped, where each file starts a new scope and each tuple forms its
own subscope.

This leads to the following common trip-ups:

### When you want to pass values to a different file, you need to declare them in the receiving file

To be able to refer to a variable, an expression needs to be able to "see" it.
This means that it must be in the same tuple or an enclosing tuple of the one
the expression is in.

If you want to pass a variable then from one file to another, the receive file
must have an "empty declaration" of that variable somewhere. For example:

    # http.gcl
    server = {
        dirname;
        www_root = '/var/www/' + dirname;
    };

And use it as follows:

    # main.gcl
    http = include 'http.gcl';
    pics_server = http.server {
        dirname = 'pics';
    };

A file behaves like a tuple, so if you need to refer to the same value a lot in
a file, you can make it a file-level parameter as well:

    # http.gcl
    port;

    server = {
        host = '0.0.0.0';
        bind = host + ':' + port;
    };

And use it as:

    # main.gcl
    https = include 'http.gcl' { port = 443 };
    pics_server = https.server {
        ...
    };

The downside of this design decision of scoping is that you need to type more,
because you need to declare all the "empty variables" that you're expecting to be
using.

On the plus side, you know *exactly* what you're referring to, and tuples that
are you are going to be mixed into can not affect the binding of your variables
in any way. See the last paragraphs of MOTIVATION.md for more information on
this.

### Copying a value from an outer scope needs the 'inherit' keyword

Let's say you want to copy a variable from an outer tuple onto a particular
tuple under the same name. Let's say you want to write something like this:

    base_speed = 3;
    motor = {
        base_speed = base_speed;  # Recursive reference
        speed = base_speed * 2;
    };

First of all, you may not need to do this! If you just wanted the variable
`speed` set to the correct value, there's no need to copy `base_speed` onto the
`motor` tuple. You can easily refer to the `base_speed` variable directly.

If you still want to copy the variable, the code as written won't work.
`base_speed` on the right refers to `base_speed` on the left, whose value is
`base_speed` on the right, which refers to `base_speed` on the right, and so on.

To solve this, you can do one of two things: rename the variable, or use the
`inherit` keyword (which copies the variable from the first outer scope that
contains it while ignoring the current scope).

So either do:

    base_speed = 3;
    bspd = base_speed;
    motor = {
        base_speed = bspd;
        speed = base_speed * 2;
    };

Or do:

    base_speed = 3;
    motor = {
        inherit base_speed;
        speed = base_speed * 2;
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
