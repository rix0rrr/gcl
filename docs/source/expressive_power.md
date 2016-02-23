Expressive Power
================

It's the eternal problem of a DSL intended for a limited purpose: such a
language then gets more and more features, to gain more and more expressive
power, until finally the language is fully generic and any computable function
can be expressed in it.

> "In a heart beat, you're Turing complete!" -- Felienne Hermans

Not by design but by accident, GCL is actually one of those Turing complete
languages. It wasn't the intention, but because of the abstractive power of
tuples, lazy evaluation and recursion, GCL actually maps pretty closely onto the
Lambda Calculus, and is therefore also Turing complete.

Having said that, you should definitely not feel encouraged to (ab)use the
Turing completeness to do calculations inside your model. That is emphatically
_not_ what GCL was intended for. This section is more of an intellectual
curiosity, and should be treated as such.

Tuples are functions
--------------------

Tuples map very nicely onto functions; they can have any number of input and
output parameters. Of course, all of this is convention. But you can see how
this would work as I define the mother of all recursive functions, the Fibonacci
function:

    fib = {
        n;
        n1 = n - 1;
        n2 = n - 2;
        value = if n == 0 then 0
           else if n == 1 then 1
           else (fib { n = n1 }).value + (fib { n = n2 }).value;
    };

    fib8 = (fib { n = 8 }).value;

And then:

    $ gcl-print fib.gcl fib8

    fib8
    21

Hooray! Arbitrary computation through recursion! 

A more elaborate example
------------------------

Any time you need a particular function, you can inject it from Python, _or_ you
could just write it directly in GCL. Need `string.join`? Got you covered:

    string_join = {
        list;
        i = 0;            # Hey, they're default arguments!
        sep = ' ';

        next_i = i + 1;
        suffix = (string_join { inherit list sep; i = next_i }).value;
        my_sep = if i > 0 then sep else '';
        
        value = if has(list, i) then my_sep + list(i) + suffix else '';
    };

    praise = (string_join { list = ['Alonzo','would','be','proud']; }).value;

We make use of the lazy evaluation property here to achieve readability by
giving names to subparts of the computation: the key `suffix` actually only
makes sense if we're not at the end of the list yet, but we can give that
calculation a name anyway. The expression will only be evaluated when we pass
the `has(list, i)` test.

Multi-way relations
------------------

Because all keys are lazily evaluated and can be overridden, we can also encode
relationships between input and output parameters in both directions. The
_caller_ of our relation tuple can then determine which value they need. For
example:

    Pythagoras = {
        a = sqrt(c * c - b * b);
        b = sqrt(c * c - a * a);
        c = sqrt(a * a + b * b);
    }

Right now we have a complete relationship between all values. Obviously, we
can't evaluate any field because that will yield an infinite recursion. But we
_can_ supply any two values to calculate the remaining one:

    (Pythagoras { a = 3; b = 4}).c    # 5

    (Pythagoras { a = 5; c = 13}).b   # 12


Inner tuples are closures
-------------------------

Just as tuples correspond to functions, nested tuples correspond to closures,
as they have a reference to the parent tuple at the moment it was evaluated.

For example, we can make a partially-applied tuple represents the capability of returning elements
from a matrix:

    Matrix = {
        matrix;

        getter = {
            x; y;
            value = matrix y x;
        };
    };

    PrintSquare = {
        getter;
        range = [0, 1, 2];

        value = [[ (getter { inherit x y }).value for x in range] for y in range];
    };

    my_matrix = Matrix {
        matrix = [
            [8, 6, 12, 11, -3],
            [20, 6, 8, 7, 7],
            [9, 83, 8, 8, 30],
            [3, 1, 20, -1, 21]
        ];
    };

    top_left = (PrintSquare { getter = my_matrix.getter }).value;

    
Let's do something silly
------------------------

Let's do something very useless: let's implement the Game of Life in GCL using
the techniques we've seen so far!

Our GCL file is going to load the current state of a board from a file and compute the next state of
the board--after applying all the GoL rules--into some output variable. If we then use a simple bash
script to pipe that output back into the input file, we can repeatedly invoke GCL to get some
animation going!

We'll make use of the fact that we can `include` JSON files directly, and that we can use `gcl2json`
to write some key back to JSON again.

Let's represent the board as an array of strings. That'll print nicely, which is
convenient because we don't have to invest a lot of effort into rendering. For example:

    [
      "............x....",
      "...x.............",
      "....x.......xxx..",
      "..xxx.......x....",
      ".............x...",
      ".................",
      ".................",
      "...x.x..........."
    ]

First we'll make a function to make ranges to iterate over.

    # (range { n = 5 }).value == [0, 1, 2, 3, 4]
    range = {
        n; i = 0;

        next_i = i + 1;
        value = if i < n then [i] + (range { i = next_i; inherit n }).value else [];
    };

Then we need a function to determine liveness. We'll expect a list of chars, either 'x' or '.', and
output another char.

    # (liveness { me = 'x'; neighbours = ['x', 'x', 'x', '.', '.', '.'] }).next == 'x'
    liveness = {
        me; neighbours;

        alive_neighbours = sum([1 for n in neighbours if n == 'x']);
        alive = (me == 'x' and 2 <= alive_neighbours and alive_neighbours <= 3)
            or (me == '.' and alive_neighbours == 3);
        next = if alive then 'x' else '.';
    };

On to the real meat! Let's find the neighbours of a cell given some coordinates:

    find_neighbours = {
        board; i; j;
        
        cells = [
            cell { x = i - 1; y = j - 1 },
            cell { x = i;     y = j - 1 },
            cell { x = i + 1; y = j - 1 },
            cell { x = i - 1; y = j     },
            cell { x = i + 1; y = j     },
            cell { x = i - 1; y = j + 1 },
            cell { x = i;     y = j + 1 },
            cell { x = i + 1; y = j + 1 }
        ];

        chars = [c.char for c in cells];

        # Helper function for accessing cells
        cell = {
            x; y;

            H = len board;
            my_y = ((H + y) % H);
            W = len (board my_y);
            char = board (my_y) ((W + x) % W);
        }
    };

Now we can simply calculate the next state of the board given an input board:

    next_board = {
        board;

        rows = (range { n = len board }).value;
        value = [(row { inherit j }).value for j in rows];

        row = {
            j;

            cols = (range { n = len board(j) }).value;
            chars = [(cell { inherit i }).value for i in cols];
            value = join(chars, '');

            cell = {
                i;
                neighbours = (find_neighbours { inherit board i j }).chars;
                me = board j i;
                value = (liveness { inherit me neighbours }).next;
            };
        };
    };

We've got everything! Now it's just a matter of tying the input and output together:

    input = {
        board = include 'board.json';
    };

    output = {
        board = (next_board { board = input.board }).value;
    };

That's it! We've got everything we need! Test whether everything is working by running:

    $ gcl2json -r output.board game_of_life.gcl output.board

That should show the following:

    [
      "....x............",
      "............x....",
      "..x.x.......xx...",
      "...xx.......x.x..",
      "...x.............",
      ".................",
      ".................",
      "................."
    ]

Hooray, it works!

For kicks and giggles, we can turn this into an animation by using `watch`, which will
run the same command over and over again and show its output:

    $ watch -n 0 'gcl2json -r output.board game_of_life.gcl output.board | tee board2.json; mv board2.json board.json'

Fun, eh? :)
