Generating library documentation
================================

If you use GCL to model complex domains, you'll want to add abstractions in the
form of new tuples to capture commonalities. You'll probably also want to share
these abstractions with your team members. But how do they figure out what kinds
of abstractions exist, and how they're supposed to be used?

For that purpose, GCL supports _generated documentation_ à la JavaDoc.

## Prerequisites

GCL comes with a tool that can extract documentation to ReSTructured text files
(`.rst`), which are intended to be used with
[Sphinx](http://www.sphinx-doc.org/). You'll need Sphinx on your machine to
generate these docs.

## Writing doc comments

Doc comments start with `#.`. The first empty line-separated block will be
treated as a title, and blocks indented with 4 spaces will be treated as code
samples. Doc comments apply to member declarations in tuples.

For example:

    #. Greeting generator
    #.
    #. This is a greeter which greets whatever name you give it.
    #.
    #. Example:
    #.
    #.     john_greeter = Greeter { 
    #.         who = 'john';
    #.     }
    Greeter = {
        #. The name of the person to greet.
        who;

        #. The greeting that will be produced.
        greeting = 'Hello ' + who;
    }

## Special documentation tags

Add tags into the doc comment to influence how the documentation is generated.
Tags are prefixed with an `@` sign and must occur by themselves on a line in the
comment block, much like JavaDoc. The tags that are currently supported are:

* `@detail`, hide this member from the documentation.
* `@hidevalue`, don't show the default value for this member in the
  documentation. By default, non-tuple values are shown.
* `@showvalue`, do show the default value for this member in the documentation.
  By default, tuple values are not shown.

## Generating the documentation

The tool that generates the documentation is `gcl-doc`. Example invocation:

    gcl-doc -o doc_dir lib/*.gcl

This generates an `index.rst` file plus a file per GCL file into the `doc_dir`
directory. You should then run sphinx on that directory (either by adding
in a `conf.py` file with some config, or by passing all interesting arguments
on the command line):

    sphinx-build -EC -Dmaster_doc=index -Dhtml_theme=classic doc_dir doc_dir/out
