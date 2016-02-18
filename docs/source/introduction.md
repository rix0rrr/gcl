Introduction
============

GCL is a declarative modeling language that can be dropped into any Python project. It supports
dictionaries with name-value pairs, all the basic types you'd expect, lists, includes, and methods
for abstraction.

Vision
------

The goal for GCL is to be a modeling language with lots of expressive power, intended to make
complex configurations extremely DRY. Behavior is not part of the goal of the language; behavior and
semantics are added by scripts that interpret the GCL model.

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

Alternatives
-----------

GCL may not be for you. These are some other popular choices that fill the same space:

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

