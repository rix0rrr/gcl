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

GPath
------

Sometimes you want to select specific values out of a big model. For that purpose, you can use
GPath, which is a hierarchical selector language to select one or more values from a model.

GPath queries look like this:

    # Select node by name
    name.name.name

    # Select multiple nodes 
    name.{name1,name2}.name

    # Select all nodes at a given level
    name.*.name

    # List indices are numbers in the path
    name.0.name

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
