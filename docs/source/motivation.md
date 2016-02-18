Why should I care about GCL?
============================

I'd like to take you through the design motivations of GCL as a configuration
language. In particular, I'm going to show what the advantages are compared to
writing JSON by hand.

## A gentle note of warning

The goal of GCL is to _reduce repetition_ in a complex JSON configuration.
However, in trying to tame that complexity, we have had to introduce _some_
syntactical and cognitive overhead. This means that if your config is tiny, the
fancy features of GCL may not be a win in your particular situation. Of course,
you can still just use it as a less-noisy, more friendly-to-type JSON variant
:).

## Whither JSON?

First off, why are we competing with JSON? JSON is not a _configuration_
format. It's a _data serialization_ format. It exists so computer programs can
exchange structured information in an unambiguous way.

However, it has a couple of things going for it, that make it a popular choice
for configuration files:

* JSON has been popularized by the web, so there are libraries to parse and
  produce it in virtually every language.
* It directly maps to an arbitrarily complex in-memory data structure. Lists,
  dicts and scalars are enough to make any type of data structure
  representation you want, in a more compact way than, say, INI files.

The advantages that JSON has are _ubiquity_ and _sufficient_ expressive power.
Not necessarily _great_ expressive power, but enough to get by.

In this document, I'll hope to sell you on the fact that GCL is equally good,
or better. (Supposing availability in your environment, obviously, which right
now means Python and some shell tools)

## Readability/writability

As noted before, JSON is a serialization format. Is has not necessarily been
specced to be convenient to write and maintain by people.

### Visual noise and pinky stress

First off, the symbols. GCL does away with a bunch of the visual noise and the
shifted characters you have to type:

~~~json
{
    "key": "value",
    "other": "value"
}
~~~

Versus:

~~~python
{
    key = 'value';
    other = 'value';
}
~~~

Looks very similar, right? The differences are:

* No quotes `"` around the keys. Fewer characters to type.
* `'` and `"` are both allowed to quote strings, but the `'` doesn't require
  holding the shift key.
* `=` is used to separate value from key instead of `:` (again, no shift key
  required).    
* `;` is used to separate fields instead of a `,`. There's no difference in
  keystrokes, but note that the `,` is _not allowed_ after the last line in
  JSON, whereas it is allowed (but optional) in GCL. This makes it less likely
  that you introduce a syntax error while adding or reordering lines.
    
These tiny differences add up to make GCL more comfortable to write. At least,
I've noticed a lot less stress on my pinkies :). And because there's less
visual noise on the quoted keys, it's easier to read as well (especially given
the syntax highlighting you can find at
[vim-gcl](http://github.com/rix0rrr/vim-gcl)).

### Comments

In a complex configuration, you're more likely that not wanting to add comments
telling the people after you why things are the way they are. JSON has not been
designed for this, and there is no way to add comments to your JSON
configuration file.

There are some hacks, such as adding a `comment` field to your dictionary,
which will then be ignored by the processing script:

~~~json
{
    "machines": 500,
    "access_rights": "pedro",
    "comment": "As many machines as we have data shards. Pedro is the guy running the analysis." 
}
~~~

In contrast, GCL supports actual comments in the file:

~~~python
{
    machines = 500;           # We have this many shards
    access_rights = 'pedro';  # In charge of the analysis
}
~~~

## GCL Extensions

At this point, you can already use GCL as a more readable JSON. All the same
value types are afforded: dictionaries, lists, strings, numbers and booleans.
As a bonus, the files are slightly easier to write.

However, GCL also affords some abstraction capabilities that make it more
expressive than just JSON:

### Includes

If you have a very large configuration file, you may want to split it up.
Includes give you this ability:

~~~python
frontend = include 'frontend.gcl';
~~~
    
The included file will always be resolved relative to the file the include is
being done from, so you don't have to worry about the script's working
directory.

### Expressions

In GCL it's possible to use expressions anywhere a value can appear. So you can
do something like:

~~~python
cores_per_machine = 4;  # This is true for all our machines

task = {
    jobs = 100;
    machines = jobs / cores_per_machine;
};
~~~
    
As you can see, it's possible to refer to keys OUTSIDE of the current scope as
well. You can also use this to build strings:

~~~python
region = 'us-east-1';

photo_bucket = { name = region + '-photos' };
mail_bucket = { name = region + '-mails' };
~~~

Or, slightly nicer to read, use the `fmt()` function to replace
`{placeholders}` inside your strings with the values of keys from the current
environment:

~~~python
region = 'us-east-1';

photo_bucket = {
    name = fmt('{region}-photos');
};
mail_bucket = {
    name = fmt '{region}-mails';  # You can leave out the ()
};
~~~

For functions of 1 argument, you can leave out the parentheses for even less
visual noise.

### Tuple composition

The big distinguishing feature of GCL, however, is _tuple composition_. If you
put 2 (or more) tuples next to each other, they will be merged together to form
one tuple. We also call this _mixing in_ a tuple. For example:

~~~python
{ a = 1 } { b = 2 }
~~~
    
Yields a combined tuple of `{ a = 1; b = 2 }`. Of course, by itself this is not
very interesting. The neat thing is that you can override values in a tuple
while composing, AND by giving names to tuples, you can make
easily-recognizable abstractions. Returning to our earlier example of `task`:

~~~python
cores_per_machine = 4;
        
# By convention, we use capitalized names for 'reusable' tuples
Task = { 
    jobs = 1;       # Default
    machines = jobs / cores_per_machine;
};
        
small_task = Task {
    jobs = 100;     # Overridden value
};
        
large_task = Task {
    jobs = 4000;    # Overridden value
};
~~~

Afterwards, `small_task.machines` is `25`, and `large_task.machines` is `1000`.
As you can see, the calculation for `machines` is inherited from `Task` each
time, evaluated with their own value for `jobs`.

In the previous example, we had a default value for `Task.jobs`, which could be
overridden (or not). It's also possible to _not_ give any value for keys we're
going to use in an expression, to force the users to fill it in:

~~~python
region;
Bucket = {
    id;
    name = region + '-' + id;
};

# Set 'id', get 'name' for free!
photo_bucket = Bucket { id = 'photos' };
mail_bucket = Bucket { id = 'mails' };

# We can still just set 'name' to bypass the the logic
music_bucket = Bucket { name = 'cool-music' };
~~~
    
In this case, we encoded the fact that a bucket name always consists of the
region and some other identifier in the `Bucket` tuple. We added another field
to fill in the part of the name that changes, and set that when we mix in the
tuple.

> Interesting note: actually, the syntax for tuple composition is exactly the
> same as for function application! You might also put parentheses around the
> second tuple, as an argument. However, the space-separation looks cleaner,
> don't you think?

> In fact, if you squint really hard at it, you might also consider that a
> tuple is a function like in normal programming languages. The declared keys
> without values are its parameters, keys with values are parameters with
> default arguments, and a composited tuple is the result of evaluating the
> function against a set of arguments.

## Principle: don't repeat yourself!

The guiding principle behind all of these mechanisms is simple, and it's
ancient: Don't Repeat Yourself! 

Every piece of policy, every (changeable) decision, should exist in only once
place. That has two advantages:

* If the rule ever needs to change, it's easy to do it in one place! If your
  naming scheme ever changes from `{region}-{id}` to
  `{component}-{region}-{id}`, you only need to make that change in one place,
  instead of search-replace through a bunch of files.
* Deviations from the norm stand out more. If every `Bucket` uses the `id =
  '...'` mechanism to have their `name` derived, _except one because its name
  needs to follow a different scheme_, you can tell at a glance that a
  particular bucket is different from the rest and you need to pay more
  attention. Ideally, that would be preceded by a comment line telling you WHY
  it's different :).

## Rule: you can always tell where a variable is coming from

I want to highlight this design decision, as it leads to a trade-off that
trades some (visible) verbosity for future (not-so-visible) safety. I want to
highlight it because, while you're writing some magic incantation, you may be
tempted to curse my name for the extra effort you have to put in. Rest assured,
there is a reason for this.

The rule is that you always want to be able to decide _what key you are
referring to in an expression from looking at a piece of code_. No client using
your tuple should be able to change the meaning of any of the keys without you
being aware beforehand.

Let me illustrate with a (silly) example involving 2 files:

~~~python
# tasklib.gcl

cpm = 4;  # Cores per machine

Task = {
    jobs = 1;
    machines = jobs / cpm;
};
~~~

~~~python
# tasks.gcl

lib = include 'tasklib.gcl';

ad_task = lib.Task {
    ads = 100000;
    jobs = ads;
    
    cpm = 0.01;  # Cost per mille
    total_cost = cpm * (ads / 1000);
};
~~~

You may see the problem already; in the instantiation of `Task.machines`, what
should the expression `cpm` refer to? You would _expect_ it to refer to the key
`cpm` from its global scope, which has a defined meaning. But all of a sudden,
someone else mixed in a key that's _also_ called `cpm`! If we were to use the
simple rule that we would take the key from the innermost tuple that has it, we
would be referring to the 'costs per mille' value instead of the 'cores per
machine' value that we intended to. 

What's more, we wouldn't be able to tell by looking at any of the pieces of
config in isolation that this could happen! And all clients would have to be
aware of all variables that exist in the enclosing scopes of a tuple, because
it cannot use the same names for any of its variables! That would lead to some
very annoying debugging sessions.

So instead, we have the rule that names in expressions will only bind to keys
that are actually _visible in the code_ from the point of view of the
expression (either in the tuple itself or one of the enclosing tuples). _New
keys_ that are mixed in later on will be ignored.

This means, slightly annoyingly but very safe, that you have to declare all
keys you're going to be expecting or using from your 'sibling' tuples.

So you've already seen this pattern, a reusable tuple expecting a parameter:

~~~python
Lyric = {
    object;
    phrase = fmt "There's a {object} in my bucket";
};

first = Lyric { object = 'hole' };
~~~

But mixing goes both ways, if a reusable tuple defines some subtuple that a
specialization might want to use:

~~~python
Lyric = {
    object;
    phrases = {
        complaint = fmt "There's a {object} in my bucket";
        salutation = 'Dear Liza';
    };
};

first = Lyric {
    object = 'hole';
    
    phrases;
    song = fmt '{phrases.complaint}, {phrases.salutation}, {phrases.salutation}, {phrases.salutation}';
    formal_letter = fmt '{phrases.salutation}, {phrases.complaint}. Regards, Henry.';
};
~~~

In this case, the right-hand tuple of `first` had to declare `phrases` as a
value-less key to bring it _into scope_, so that it can refer to it in
expressions. Without this declaration, it wouldn't know to get the value of
`phrases` from a tuple that's mixed-in on the side, but would have to go and
look for it in an enclosing scope (where no such key exists).

## Influences on script interpreting GCL models

GCL affords a lot of expressivity, allowing to you keep config files textually
simple, while (in essence) "generating" big flat objects when all of the
abstractions are evaluated away.

This has an influence on the scripts that you write to parse your config. Since
you can a do bunch of stuff in GCL, the script doesn't have to implement
mechanisms to keep configs maintainble by humans. 

* For example, to have both generic configs with "template holes" and then a
  separate set of values that fill the holes with string substitution.
* Also, the script writer doesn't need to think of what would be useful
  abstractions (as the config writer can notice them while writing the config,
  and abstract them out immediately);
* And the script writer doesn't need to invent additional mechanisms to
  override the default abstractions when users inevitably run into a situation
  that doesn't fit the mold.

Instead, the script will typically operate on arrays of big fat, flat objects
that have every value needed directly available in the dictionary to do
whatever it is the script needs doing. This makes writing the script at lot
easier.

As an example of something else the script writer doesn't have to deal with:
GCL contain a tool called `gcl-print` to print an exploded config file. Helpful
to config writers.

The flip side of the possible complexity is that GCL configs are actually code,
and should be treated as such [^1]. That means that naming and documenting all
the convenient "abstractions" are just as important in the config as they would
be in regular code. 

A schema language, to assist with this, will be forthcoming soon. Good ideas
are very welcome!


[^1]: But then again, I'm of the opinion that ALL config should be treated as part of code. JSON files and INI files have just as much of an implicit contract with their interpreting script as GCL, which should be documented just as well.

