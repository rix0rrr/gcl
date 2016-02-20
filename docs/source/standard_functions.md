Functions in GCL
================

This document contains a list of all the functions that come in the standard distribution of GCL.
They're defined in `gcl/functions.py`.

## List of standard functions

### `fmt(format_string, [env])`

Does string substitution on a format string, given an environment. `env` should be a
mapping-like object such as a dict or a tuple. If `env` is not specified, the current
tuple is used as the environment.

Example:

    host = "x";
    port = 1234;
    address = fmt '{host}:{port}';

### `compose_all(list_of_tuples)`

Returns the composition of a variable list of tuples.

### `eager(tuple)`

Turn a lazy GCL tuple into a dict. This eagerly evaluates all keys, and forces the object to be
complete. This will also force a schema evaluation of all keys.

### `flatten(list_of_lists)`

Flatten a list of lists into a single list.

### `has(tuple, key)`

Return whether a given tuple has a key, and the key has a value.

### `join(list, [separator])` 

Combine a list of string using by separator (defaults to a single space if not specified).

### `path_join(str, str, [str, [...]])`

Call Python's `os.path.join` to build a complete path from parts.

### `split(str, [separator])`

Split a string on a separator (defaults to a single space if omitted).

### `sorted(list)`

Return `list` in sorted order.

### `sum(list)`

Sums a list of numbers.

## Custom functions

You can define new functions for use inside GCL (in fact, you can bind arbitrary values to any
identifier) by passing in new bindings for the initial environment, using the keyword argument `env`
when calling `gcl.read`:

    import gcl
    import string

    my_bindings = { 'upper': string.upper }

    object = gcl.loads('yell = upper "hello"', env=my_bindings)
    print(object['yell'])

