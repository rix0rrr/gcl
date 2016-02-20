Model Schemas
=============

When working in a larger project, when multiple people are involved, as many
automatic sanity checks should be added to a code base as possible. Static
typing, or its cousin in data specifications, schemas, are one of the tools we
have to reduce mistakes and frustration.

Especially when the GCL is going to be processed in a dynamically typed language
like Python where errors can be delayed for a long time (and writing explicit
type checks is cumbersome and often omitted) validating the input data as
quickly as possible makes sense.

## Use cases

Typically, we want to guard against the following two scenarios:

* Missing (required) fields; and
* Values of incorrect types provided.

Schemas annotations are also used for automatic extraction of relevant data: a common use of GCL is
as a preprocessor for generating JSON. However, typically you'll have additional fields in your GCL
tuples that are used for abstraction, which you don't want to have end up in the JSON. We'll use the
schema to extract only the relevant fields from your objects.

## Specifying a schema

Schemas are specified in GCL itself. Typically, you'd put them on tuples are
that are designed to be mixed in with other tuples, to prevent them from being
misused. For example, you can specify the types of input keys, or force
downstream users to provide particular keys.

### Scalars

Schema specification for scalars looks like this:

    Human = {
        hands : required int;
        fingers = 5 * hands;
    }

    lancelot = Human {
        hands = 2;
    }

When mixing the tuple `Human` into another tuple, a value for `hands` must be
specified, and it must be an integer. Failing to provide that key will throw an
error when the tuple is accessed. The following built-in scalar types are
defined:

    string
    int
    float
    bool
    null

> Note: Since `:` is a valid part of an identifier, there must be a space
> between the identifier and the `:` of the schema specification.

### Lists

Specifying that a key must be a list looks is done by specifying an example type
using `[]`, optionally containing a schema for the elements:

    Human = {
        names : required [string];
        inventory : [];
    };

    lancelot = Human {
        names = ['Sir', 'Lancelot', 'du', 'Lac']; 
        inventory = ['armor', { type = 'cat'; name = 'Peewee' }];
    };

### Tuples

To specify that a key must be a tuple (even a tuple with a specific type),
specify an example tuple with the expected schema, OR even just refer to an
exising tuple with the intended schema:

    Human = {
        name : required string;

        father : Human;
        mother : Human;
        children : [Human];
    };

    lancelot = Human {
        father = { name = 'King Bang' };
        children = [galahad];
    };

    galahad = Human {
        father = lancelot;
        mother = { name = 'Elaine' };
    };

## Controlling key visibility on exports

Sometimes you're building a model that you want to export to another program. A
typical example would be to use GCL as a preprocessor to have a deep model with
lots of abstraction, "compiling down" to a flat list of easy to process objects
stored in a JSON file (for example for a single page web app).

Some keys in tuples will be intended for the final output, and some are just to
input keys for the abstraction mechanisms. By marking a key as `private`, it
will not be exported when calling `to_python`.

For example, given the model from before:

    Human = {
        hands : private required int;
        fingers = 5 * hands;
    };

    lancelot = Human {
        hands = 2;
    };

`hands` is only an implementation detail. We're actually only interested in the
number of fingers that Lancelot has. By running `gcl2json` on this model:

    $ gcl2json knights.gcl lancelot

    {"lancelot": {"fingers": 10}}

`hands` is nowhere to be found!
