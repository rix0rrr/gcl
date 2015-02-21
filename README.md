
# Why not JSON?

- No comments
- No expressions
- No abstraction (tuple composition, includes...)
- All the double quotes!

# Why not XML?

- Puh-leez.



{ 
  base = {
    name;
    hello = 'hello ' + name;
  }

  mine = base { name = 'hoi' }
}


{ 
  base = {
    hello = 'hello';
  }

  mine = base {
    tak = base.hello;
  }
}
