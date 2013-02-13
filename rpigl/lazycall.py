"""
Some utility functions to perform actions lazily upon the first
call to a function.
"""


import functools

class lazycall(object):
    """Wrap a callable which returns itself callables,
    so that it is executed lazily, i.e. the wrapped callable
    is only called (once) on the first call to the returned callable.

    From a user point of view it is as if callable is called with args, except that
    any side effects are only visible upon the first call the the result.
    """
    
    __slots__ = ["callable", "args", "f"]

    def __init__(self, callable, *args):
         self.callable = callable
         self.args = args
         self.f = None

    def __call__(self, *args2):
        f = self.f
        if f is None:
            f = self.callable(*self.args)
            self.f = f
            del self.callable
            del self.args
        return f(*args2)


def lazycallable(wrapped):
    return functools.partial(lazycall, wrapped)


class LazyAttr(object):
    """Objects of this class create attributes automatically on first access.
    """
    def __init__(self, callable):
        self._callable = callable
        self._all = []

    def __getattr__(self, name):
        value = self._callable(name)
        self.__dict__[name] = value
        self._all.append(value)
        return value

    def __len__(self):
        return len(self._all)

    def __iter__(self):
        return iter(self._all)
