Command Line Tools
==================

## gcl-print

While editing GCL model files, it's helpful to look at the results of evaluating
that model. `gcl-print` can evaluate and print a (subset of) a model.

    $ gcl-print python.gcl

    +- blue              => '#0000ff'
    +- knight
    |  +- armor          => 'chain_mail'
    +- lancelot
    |  +- armor          => 'chain_mail'
    |  +- favorite_color => '#0000ff'
    |  +- helmet         => <python.gcl:10: while evaluating 'armor' in '    helmet = armor;', Unbound variable: 'armor'>

`gcl-print` accepts [GPath](loading_and_using.html#gpath) selectors.

## gcl2json

GCL can be used as a preprocessor for a complicated JSON model, which can then
be processed using more standard tools. `gcl2json` loads GCL model and spits it
out in JSON format.

    $ gcl2json python.gcl
    
    {"blue": "#0000ff", "knight": {"armor": "chain_mail"}, "lancelot": {"armor": "chain_mail", "favorite_color": "#0000ff"}}

`gcl2json` accepts [GPath](loading_and_using.html#gpath) selectors.
