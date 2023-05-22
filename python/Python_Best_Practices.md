# Python Best Practices

## Exception Handle

Errors detected during execution are called `exceptions` and are not unconditionally fatal

### Key Objectives of Exception Handle**
* In a large long running application, Exception handling allows us to define multiple conditions that we can skip and deal with later
* We can log the issue encoutered without interrupt the program
* Avoid complexity of conditional statement

### Common exceptions
* `Exception`: Base class for all exceptions. If you are not sure about which exception may occur, you can use the base class. It will handle all of them
* `ZeroDivisionError`: It is raised when you try to divide a number by zero
* `ImportError`: It is raised when you try to import the library that is not installed or you have provided the wrong name
* `IndexError`: Raised when an index is not found in a sequence. For example, if the length of the list is 10 and you are trying to access the 11th index from that list, then you will get this error
* `IndentationError`: Raised when indentation is not specified properly
* `ValueError`: Raised when the built-in function for a data type has the valid type of arguments, but the arguments have invalid values specified

### Common use cases
1. `try ... except`

    * If an exception occurs during execution of the `try` clause, the rest of the clause is skipped. Then, if its type matches the `exception` named after the except keyword, the except clause is executed, and then execution continues after the try/except block.
    * If an exception occurs which does not match the exception named in the except clause, it is passed on to outer try statements; if no handler is found, it is an unhandled exception and execution stops with a message as shown above.

        ```python
        try:
            return a/b
        # if the error occurs, handle it !!
        except ZeroDivisionError:
            print("Cannot divide by Zero!!")
        ```

2. `try ... except ... else`

   * The optional `else` clause, which, when present, must follow all except clauses. It is useful for code that must be executed if the try clause does not raise an exception. For example:

        ```python
        try:
            print(a/b)
        # if the error occurs, handle it !!
        except ZeroDivisionError:
            print("Cannot divide by Zero!!")
        # if no error occurs		
        else:
            print("No Error occured!!")
        ```
   3. `try ... except ... finally`
    * The `finally` clause always get executed whether the program gets any of the exceptions or not.
        ```python
        try:
            print(a/b)
        # if the error occurs, handle it !!
        except ZeroDivisionError:
            print("Cannot divide by Zero!!")
        else:
            print("No Error occured!!")
        finally:
            print('Value of a', a, 'and b', b)
        ```


## Data Model Methods `__function__` (Dunda methods) (Protocol oriented Language)

A Common python design pattern: If you want to implement some custom behavior on a python object, we do it by implement some `__function__`  (dungda method) for that top-level behavior 

* Examples:
    ```python
    # To allow initialization of class parameter to produce more succinct code
    __init__()

    # To create more readable class representation
    __repr__
    def __repr__(self): return f'myfunction({self.var})'

    # Custom add x+y
    __add__

    # function call expression myfunc()
    __call__
    ```
* [Reference](https://docs.python.org/3/reference/datamodel.html)
	 

## Meta classes
Python is a protocol oriented langauge, any python language does in execution context (building class, executing class, import module), one can hook into them.

```python
# To Check whether a class function or property exist in base class before inherit it
hasattri()

# Metaclass allow checks in base class to ensure the correct usage of it. 
class BaseMeta(type):
    def __new__(cls, name, bases, body):
        If not 'bar' in body:
            Raise TypeError "Bad class"
        return super().__new__(cls, bases, body)
			
class Base(metaclass=BaseMeta):
    def foo(self0:
        return self.bar()

# Alternative to metclass, you can use __init__subclass()
def __init__subclass__(cls, *a, **kw):
    print('init_subclass', a, kw)
    Return super().__init_subclass__(*a, **kw)
			
```	

## Python decorator

One can access many default property of python function object (since python is a dynamic language, it does not turn function/class to an bits first such as in C++ or Java). It is a runtime object

If I have a function `F`:
* `F.__name__` : function name
* `F.__module__`: where F is defined in (module) 
* `F.__default__`: function default argument value
* `F.__code__.co_code`: what the binary code
* `F.__code__.co_argument`: what the argument

The design pattern with decorator is:
* How to write a simplest thing to achieve the same purpose?
* When things getting complex, how to linearly extent it without minimum re-writing
* **Solution**: Taking a function and wrap it with some behavior to around it.

A decorator is a syntax that wrap a behavior around a function (it is dynamically constructed since python is a run-time language)

* Example 1:

    ```python
    from functools import wraps
    def my_decorator(f):
        @wraps(f)
        def wrapper(*args, **kwds):
            print('Calling decorated function')
            return f(*args, **kwds)
        return wrapper

    @my_decorator
    def example():
        """Docstring"""
        print('Called example function')
    ```

    excution output

    ```bash 
    >>>example()
    Calling decorated function
    Called example function
    >>> example.__name__
    'example'
    >>> example.__doc__
    'Docstring'
    ```

* Example 2:
    ```python
    def annealer(f):
        def _inner(start, end): return partial(f, start, end)
        return _inner

    @annealer
    def sched_lin(start, end, pos): return start + pos*(end-start)
    ```

### Commonly used decorators

* `@classmethod`
  * receives the class as an implicit first argument `cls`, just like an instance method receives the instance  
  * A class method is a method that is bound to the class and not the object of the class.
  * They have the access to the state of the class as it takes a class parameter that points to the class and not the object instance.
  * It can modify a class state that would apply across all the instances of the class. For example, it can modify a class variable that will be applicable to all the instances.
  * You can call a class method to apply a different object of the same class
* `@static_method` 
  * a class function that does not receive implicit `self` argument
  * it bound to class but not bound an class object
  * it belongs to a class for perform certain function which does not alter class objects
* `@property` : see details below 

## Python Generator
 
`Eager`: A function irrespective how many time it computes, it consumes the same amount of time and memory which is wastful

A better behavior is iterate over, give one value at time, so one can process it right away instead of waiting for the full data, it require no memory storage. If we ought to define a function, we can use the following data model for implementation
* `__iter__`
* `__next__`


But python makes it simple with generator syntax: the same syntax of iterator with with `yield`

One interesting behavior of generator is that it not only give 1 data back with yield, it also give the control back to user before the next data. We can interleave generator and user code to data which can enable co-routines. 

We can use generator to enforce sequences and steps

```python
class api():
    First_step()
    yeild
    Second_step()
    yield
    last_step()
```

## Context manager 

We have some setup action and some tear down action, we want some behavior in between

```python
with open('text.py') as f:
    Pass
```

Behind the scene implement using `__enter__` and `__exit__`

```python
class template:
    def __init__(self):
    def __enter__(self): some action upn enter
    def __exit__(self): some action upon exit
```

Create custom context functions
```python
from contextlib import contextmanager

@contextmanager
def managed_resource(*args, **kwds):
    # Code to acquire resource, e.g.:
    resource = acquire_resource(*args, **kwds)
    try:
        yield resource
    finally:
        # Code to release resource, e.g.:
        release_resource(resource)
```

```bash
>>> with managed_resource(timeout=3600) as resource:
...     # Resource is released at the end of this block,
...     # even if code in the block raises an exception
```


## Put the above concept together		

The better way is to use generator to ensure exit after enter (we can use python contextlib module)

```python
def template():
    action
    Yield
    action

class template:
    def __init__(self):
    def __enter__(self): 
        self.gen = template()
        mext(self.gen)
    def __exit__(self):
        next(self.gen, None)
```

We can further improve it by wrapping this behavior around and create a decorator so we can use `decorator`, `generator`, `context manager` all together

Here is a combined example:

```python
from sqlite3 import connect
from contextlib import contextmanager

@contextmanager
def template(cur):
    cur.execute('create table points(x, int, y int)')
    try:
        Yield
    finally:
        cur.execute('drop table points')

with connect('test.db') as conn:
    cur = conn.cursor()
    with template(cur):
        cur.execute(…)
        cur.execute(…)
        cur.execute(…)
        cur.execute(…)
```

## Property

Python class ofen contains class attributes that needs to be accessible through the instance, class or both. If you expose those attrbutes to user then they becomes part of API of the class. The problem rise when user need to mutate attributes, i.e. change the internal class attributes. In other lauguage, one needs ot provde `getter` and `setter` methods to enable accessor and mutator function

In addition to `getter` and `setter` methods, (python does not have private, protected and public access to attributes and method, using python convention of prefixing `_` indicates it is a non-public attribute or method), the pythonic approach is turn attributes into **properties** to allow you to change the underlying implementation of attributes without changing the public API.

Properties allow us to create methods that behavior like attributes. We can turn a class internal variable into properties, one can continue access them as attribute and we can create an underlying method holding it that will allow us to modify their internal implementation and perform action on it before user try to access and mututate it.

The main advantage of Python properties is that they allow you to expose your attributes as part of your public API. If you ever need to change the underlying implementation, then you can turn the attribute into a property at any time without much pain.

We can use `property` either as function or as a decorator (functions that take another function as an argument and return a new function with added functionality. With a decorator, you can attach pre- and post-processing operations to an existing function)


* Property as function
    ```python
    property(fget=None, fset=None, fdel=None, doc=None)
    """
    :params fget: Function that returns the value of the managed attribute
    :params fset: Function that allows you to set the value of the managed attribute
    :params fdel: Function to define how the managed attribute handles deletion
    :params doc: String representing the property’s docstring
    :return: managed attribute
    """
    ```

  * If we access `obj.attr`, python auto call `fget()`. 
  * If we run `obj.attr = value`, python calls `fset()`. 
  * If we run `del obj.attr`, python call `fdel()`
  * If we run `help(obj)`, python call `doc`
  * Example:
    ```python
    class Circle:
        def __init__(self, radius):
            self._radius = radius

        def _get_radius(self):
            print("Get radius")
            return self._radius

        def _set_radius(self, value):
            print("Set radius")
            self._radius = value

        def _del_radius(self):
            print("Delete radius")
            del self._radius

        radius = property(
            fget=_get_radius,
            fset=_set_radius,
            fdel=_del_radius,
            doc="The radius property."
        )
    ```

* As decorator
  * `@property` implement `getter` logic
  * `@obj.setter` impement `setter` logic
  * `@obj.deleter` impement `deleter` logic
  * Docstring is part of the `getter` logic
  ```python
  class Circle:
    def __init__(self, radius):
        self._radius = radius

    @property
    def radius(self):
        """The radius property."""
        print("Get radius")
        return self._radius

    @radius.setter
    def radius(self, value):
        print("Set radius")
        self._radius = value

    @radius.deleter
    def radius(self):
        print("Delete radius")
        del self._radius
  ```
  * Summary:
    * The @property decorator must decorate the getter method.  
    * The docstring must go in the getter method.
    * The setter and deleter methods must be decorated with the name of the getter method plus .setter and .deleter, respectively.

In general, you should avoid turning attributes that don’t require extra processing into properties. Using properties in those situations can make your code: 

* Unnecessarily verbose 
* Confusing to other developers
* Slower than code based on regular attributes

Unless you need something more than bare attribute access, don’t write properties. They’re a waste of CPU time, and more importantly, they’re a waste of your time.

### Read-only
If we do not define a `setter` method, the default `.__set__()` will raise an `AttributeError` if user try to assign a value to the attribute. Or we can add `setter` that raise exception explicitly

```python
class Point:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

class WriteCoordinateError(Exception):
    pass

class Point_v2:
    def __init__(self, x, y):
        self._x = x
        self._y = y

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        raise WriteCoordinateError("x coordinate is read-only")

    @property
    def y(self):
        return self._y

    @y.setter
    def y(self, value):
        raise WriteCoordinateError("y coordinate is read-only")
```

### Read-write
Define `setter` method to enable write

```python
import math

class Circle:
    def __init__(self, radius):
        self.radius = radius

    @property
    def radius(self):
        return self._radius

    @radius.setter
    def radius(self, value):
        self._radius = float(value)

    @property
    def diameter(self):
        return self.radius * 2

    @diameter.setter
    def diameter(self, value):
        self.radius = value / 2
```

## Partial Function
Partial functions allow us to fix a certain number of arguments of a function and generate a new function. Partial functions can be used to derive specialized functions from general functions and therefore help us to reuse our code.

```python
from functools import partial
  
# A normal function
def add(a, b, c):
    return 100 * a + 10 * b + c
  
# A partial function with b = 1 and c = 2
add_part = partial(add, c = 2, b = 1)
  
# Calling partial function
print(add_part(3))
```