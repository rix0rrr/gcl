Scoping rules and their consequences
====================================

This section describes the scoping rules that GCL is built on, the design decisions that have been
made and the consequences of these decisions. At first glance, GCL might require you to type "too
much", or ask you to type things in weird places during tuple composition, but this has been done
for good reason: to keep GCL models easy to reason about, even when they grow large.

Lexical Scope
-------------

References in GCL are lexically scoped, where each file starts a new scope and each tuple forms its
own subscope.

"Lexically scoped" means that only variables that are declared in the current tuple, or any of
its parent tuples, are visible in expressions.

For example:

    blue = '#0000ff';

    knight = {
        armor = 'chain_mail';
    };

    lancelot = knight {
        favorite_color = blue;  # Visible

        helmet = armor;  # Not visible!
    };
  
Note that in the `lancelot` tuple, the assignment of `armor` to `helmet` will fail, because even
though a value for `armor` is mixed in from the `knight` tuple, that value is not _lexically_
visible!

Behaving like this is the only way to avoid "spooky action at a distance". Let's assume GCL
did _not_ behave this way, and we would always take the value from the current closest enclosing
tuple. Then what would happen if someone else, a colleage, inadvertently added a key named `blue`
onto the `knight` tuple?

    blue = '#0000ff';

    knight = {
        armor = 'chain_mail';
        blue = 'smurf';
    };

    lancelot = knight {
        favorite_color = blue;  # Whoops! I wanted '#0000ff' but I got 'smurf'!
    };

Especially when the tuples invoved are far apart, such as in different files, these effects become
nearly impossible to see and debug. Lexical scoping prevents that.

Valueless keys are input parameters
-----------------------------------

To be able to refer to a variable, an expression needs to be able to "see" it.
This means that it must be in the same tuple or an enclosing tuple of the one
the expression is in.

If you want to pass a variable then from one tuple to another, or one file to another, the receive
file must have an "empty declaration" of that variable somewhere. For example:

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

In this example, we make `https` an instantiation of the `http` library with a variable pre-bound.
This is common pattern in large GCL models, and can be thought of as "partial application" of a
collection of tuples.

As you can see, the downside of the design decision of lexical scoping is that you need to type
more, because you need to declare all the "empty variables" that you're expecting to be using. On
the plus side, you know *exactly* what you're referring to, and tuples that are you are going to be
mixed into can not affect the binding of your variables in any way.

The 'inherit' keyword
---------------------

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
