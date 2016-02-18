Loading and using GCL files
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

Bindings variables from the script
----------------------------------

Add bindings to the top-level scope from the evaluating script by passing a dictionary with bindings
to the `load` function:

    gcl.load('file.gcl', env={'swallow': 'unladen'})

This is also how you add custom functions to the model.
