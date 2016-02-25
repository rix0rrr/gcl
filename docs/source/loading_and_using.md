Using GCL files from Python
===========================

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

`to_python` will respect the visibility of keys as specified in the [schema](schemas.html).

Bindings variables from the script
----------------------------------

Add bindings to the top-level scope from the evaluating script by passing a dictionary with bindings
to the `load` function:

    gcl.load('file.gcl', env={'swallow': 'unladen'})

This is also how you add custom functions to the model.

Finding nodes by location: GPath
--------------------------------

Sometimes you want to select specific values out of a big model. For that purpose, you can use
GPath, which is a hierarchical selector language to select one or more values from a model.

GPath queries look like this:

    # Select node by name
    name.name.name

    # Select multiple nodes 
    name.{name1,name2}.name

    # Select all nodes at a given level
    name.*.name

    # List indices are numbers in the path between square brackets
    name.[0].name

The [command-line tools](command_line_tools.html) use GPath for selecting nodes from the model.
Using GPath in your own script looks like this:

    import gcl
    from gcl import query

    model = gcl.load('python.gcl')

    q = query.GPath([
        '*.favorite_color',
        'lancelot',
        ])

    results = q.select(model)

    # A list of all values found
    print(results.values())

    # A list of (path, value) tuples of all values found
    print(results.paths_values())

    # A deep copy of all selected values into a dict
    print(results.deep())


Finding nodes by criteria: TupleFinder
--------------------------------------

For some reason, I keep on using GCL in scripts where I have big collection of
"things", and I want to do an action to each particular thing. Since this is a
common usage pattern, there are some standard classes to help with that.

You have option to search the entire model for all tuples that have a particular
key (`HasKeyCondition`, and another useful pattern is to do an additional level
of dispatch on the value of that key), or for all tuples (or elements) that are
in lists with particular key names anywhere in the model (`InListCondition`).

If the found tuples contain references to one another, the `TupleFinder` even
has the ability to order tuples by dependency, so that tuples that are depended
upon come earlier in the list than the tuples that depend on them (unless there
is a cyclic reference).

    import gcl
    from gcl import query

    obj = gcl.load(model.gcl')

    finder = query.TupleFinder(query.HasKeyCondition('type', search_lists=True))
    finder.find(obj)

    # All matched objects are now in finder.unordered
    print(finder.unordered))

    # We'll do an ordering based on dependencies
    finder.order()

    if finder.has_recursive_dependency():
        print('Some nodes have a recursive dependency!')
        print(finder.find_recursive_dependency())
    else:
        # Nodes got moved to finder.ordered
        print(finder.ordered)


Thread safety
-------------

Currently, GCL evaluation is _not_ thread safe. Unfortunately we need to store some global state to
track evaluations so we can detect infinite recursion inside the model.

Let me know if this poses a problem for you, we can make it into threadlocals without much issue.
