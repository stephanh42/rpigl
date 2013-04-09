"""
Load OpenGL functions.
Not usually called directly but invoked indirectly when importing gles2.
"""

import ctypes
import ctypes.util
import functools
import platform
from .lazycall import lazycall, lazycallable


class GLError(Exception):
    """An OpenGL error condition."""
    pass

if platform.system() == "Windows":
    GLFUNCTYPE = ctypes.WINFUNCTYPE
else:
    GLFUNCTYPE = ctypes.CFUNCTYPE


def load_with_fallback(loader, lib, name):
    proc = loader(name)
    if proc is None:
        return (name, lib)
    else:
        return proc


@lazycall
def raw_load_gl_proc():
    system = platform.system()
    if system == "Linux":
        from . import rpi_egl
        if rpi_egl.is_raspberry_pi:
            gl_library = rpi_egl.gles_lib
            return lambda name: (name, gl_library)
        else:
            gl_library = ctypes.CDLL(ctypes.util.find_library("GL"))
            return (GLFUNCTYPE(ctypes.c_void_p, ctypes.c_char_p))(("glXGetProcAddressARB", gl_library))
    elif system == "Windows":
        gl_library = ctypes.WinDLL(ctypes.util.find_library("OpenGL32"))
        loader = (GLFUNCTYPE(ctypes.c_void_p, ctypes.c_char_p))(("wglGetProcAddress", gl_library))
        return functools.partial(load_with_fallback, loader, gl_library)
    else:
        raise GLError("Do not know how to load OpenGL library on platform %s" % system)


def check_gl_error(name, result, func, args):
    err = glGetError()
    if err == GL_NO_ERROR:
        return result
    elif err == GL_INVALID_ENUM:
        msg = "An unacceptable value is specified for an enumerated argument."
    elif err == GL_INVALID_VALUE:
        msg = "A numeric argument is out of range."
    elif err == GL_INVALID_OPERATION:
        msg = "The specified operation is not allowed in the current state."
    elif err == GL_INVALID_FRAMEBUFFER_OPERATION:
        msg = "The framebuffer object is not complete."
    elif err == GL_OUT_OF_MEMORY:
        msg = "There is not enough memory left to execute the command."
    else:
        msg = "%d" % err
    raise GLError("OpenGL error in %s(%s): %s" % (name.decode(), ", ".join([str(a) for a in args]), msg))


def _load_gl_proc_helper(name, restype, argtypes):
    raw_proc = raw_load_gl_proc(name)
    if raw_proc is None:
        return None
    prototype = GLFUNCTYPE(restype, *argtypes)
    try:
        return prototype(raw_proc)
    except AttributeError:
        return None


@lazycallable
def load_gl_proc(name, restype, argtypes):
    proc = _load_gl_proc_helper(name, restype, argtypes)
    if proc is None and name[-1:] == b"f":
        argtypes = [ctypes.c_double if t == ctypes.c_float else t for t in argtypes]
        proc = _load_gl_proc_helper(name[:-1], restype, argtypes)
    if proc is None:
        raise GLError("OpenGL procedure not available: %s" % name)
    if name != b"glGetError":
        proc.errcheck = functools.partial(check_gl_error, name)
    return proc



GL_NO_ERROR = 0
GL_INVALID_ENUM = 0x0500
GL_INVALID_VALUE = 0x0501
GL_INVALID_OPERATION = 0x0502
GL_STACK_OVERFLOW = 0x0503
GL_STACK_UNDERFLOW = 0x0504
GL_OUT_OF_MEMORY = 0x0505

glGetError = load_gl_proc(b"glGetError", ctypes.c_uint, ())
