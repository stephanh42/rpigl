"""
Create rpigl/gles2.py file by parsing files in the specfiles subdirectory.
Run this on the Raspberry Pi; it uses the actual libGLESv2.so to decide
which functions to include.
"""

import string
import re
import sys
import ctypes
import ctypes.util

gles_path = ctypes.util.find_library("GLESv2")
gles_lib = ctypes.CDLL(gles_path)

def has_function(name):
    try:
        getattr(gles_lib, name)
        return True
    except AttributeError:
        return False


ctypes = {
  "BooleanPointer" : "POINTER(ctypes.c_ubyte)",
  "CharPointer" : "c_char_p",
  "CheckedFloat32" : "c_float",
  "CheckedInt32" : "c_long",
  "ClampedFloat32" : "c_float",
  "ClampedFloat64" : "c_double",
  "ConstCharPointer" : "c_char_p",
  "ConstFloat32" : "c_float",
  "ConstUInt32" : "c_ulong",
  "ConstVoidPointer" : "c_void_p",
  "Float32" : "c_float",
  "Float64" : "c_double",
#  "FragmentLightModelParameterSGIX" : "FragmentLightModelParameterSGIX",
#  "GLDEBUGPROC" : "GLDEBUGPROC",
#  "GLDEBUGPROCAMD" : "GLDEBUGPROCAMD",
#  "GLDEBUGPROCARB" : "GLDEBUGPROCARB",
  "GLbitfield" : "c_uint",
  "GLboolean" : "c_ubyte",
  "GLchar" : "c_char",
  "GLcharARB" : "c_char",
  "GLdouble" : "c_double",
  "GLenum" : "c_uint",
  "GLfloat" : "c_float",
#  "GLhandleARB" : "GLhandleARB",
  "GLint" : "c_int",
  "GLintptr" : "c_size_t",
  "GLintptrARB" : "c_size_t",
  "GLshort" : "c_short",
  "GLsizei" : "c_size_t",
  "GLsizeiptr" : "c_ssize_t",
  "GLsizeiptrARB" : "c_ssize_t",
#  "GLsync" : "GLsync",
  "GLbyte" : "c_byte",
  "GLubyte" : "c_ubyte",
  "GLuint" : "c_uint",
  "GLushort" : "c_ushort",
#  "GLvdpauSurfaceNV" : "GLvdpauSurfaceNV",
  "GLvoid" : "c_void",
#  "Half16NV" : "Half16NV",
  "Int16" : "c_int16",
  "Int32" : "c_int32",
  "Int64" : "c_int64",
  "Int8" : "c_int8",
#  "MeshMode2" : "MeshMode2",
  "UInt16" : "c_uint16",
  "UInt32" : "c_uint32",
  "UInt64" : "c_uint64",
  "UInt64EXT" : "c_uint64",
  "UInt8" : "c_uint8",
  "VoidPointer" : "c_void_p",
  "charPointerARB" : "c_char_p",
#  "cl_context" : "cl_context",
#  "cl_event" : "cl_event",
  "void" : "c_void",

  "String" : "c_char_p"
}

def make_pointer(ctype):
    if ctype == "c_void":
        return "c_void_p"
    else:
        return "POINTER(ctypes.%s)" % ctype


def remove_comment(line):
    n = line.find("#")
    if n>= 0:
        return line[:n]
    else:
        return line

extensions = frozenset(("3DFX", "AMD", "APPLE", "ARB", "ARM", "ATI", "EXT", "HP", "IBM", "INTEL", "KHR", "MESA", "MESAX", "NV", "OES", "OML", "QCOM", "SGI", "SGIS", "SGIX", "SUN", "SUNX"))

def parse_enums(f):
    re_enum = re.compile(r"^\s*([A-Z0-9_]+)\s*=\s*([0-9A-Fx]+)\s*$")
    enums = []
    for line in f:
        line = remove_comment(line)
        mo = re_enum.match(line)
        if mo is not None:
            name, value = mo.groups()
            ext = name.split('_')[-1]
            if ext not in extensions:
                enums.append((name, value))
    return enums


def parse_type_map(f):
    re_mapping = re.compile(r"([a-zA-Z_]+),\*,\*,\s+([a-zA-Z_]+),\*,\*")
    type_map = {} 
    for line in f:
        line = remove_comment(line)
        mo = re_mapping.match(line)
        if mo is not None:
           type_map[mo.group(1)] = mo.group(2)
    return type_map

type_map = parse_type_map(open("specfiles/gl.tm", "r"))

extension_re = re.compile("[A-Z][A-Z]+$")

def get_extension(name):
     mo = extension_re.search(name)
     if mo is not None:
          return mo.group(0)
     else:
          return ""

    
class GLFunction:
    retval = "void"
    valid = True
    deprecated = False
    version = "1.0"

    def __init__(self, name, args):
        self.name = name
        self.args = args
        self.extension = get_extension(name)
        self.argtypes = {}

    def __str__(self):
        result = self.retval + " " + self.name + "(" + ", ".join([str(self.argtypes.get(a, "???")) + " " + a for a in self.args]) + ")"
        if not self.valid:
            result = "# "  + result
        return result

    def write_loading(self, f):
        retval = self.retval
        if retval == "c_void":
            retval = "None"
        else:
            retval = "ctypes." + retval
        f.write("_%s = load_gl_proc(b\"%s\", %s, (%s))\n" % (self.name, self.name, retval,
             " ".join(["ctypes.%s," % self.argtypes[a] for a in self.args])))
        args = ", ".join(self.args)
        f.write("def %s(%s):\n    return _%s(%s)\n" % (self.name, args, self.name, args))


def parse_gl_spec(gl_spec):
    re_header = re.compile(r"([a-zA-Z0-9]+)\(([^)]*)\)")
    current_function = None
    functions = []
    for line in gl_spec:
        line = remove_comment(line)
        mo = re_header.match(line)
        if mo is not None:
            name = "gl" + mo.group(1)
            args = [s.strip() for s in mo.group(2).split(",")]
            args = [arg for arg in args if arg]
            current_function = GLFunction(name, args)
            functions.append(current_function)
            continue
        if current_function is None:
            continue
        line = line.split()
        if len(line) == 0:
            continue
        if line[0] == "return":
            retval = line[1]
            retval = type_map.get(retval, retval)
            if retval in ctypes:
                ctype = ctypes[retval]
                current_function.retval = ctype
            else:
                current_function.retval = "???" + retval
                current_function.valid = False
        elif line[0] == "param":
            base_type = line[2]
            base_type = type_map.get(base_type, base_type)
            if base_type in ctypes:
                ctype = ctypes[base_type]
            else:
                current_function.valid = False
                ctype = "???" + base_type
            if line[4] != "value":
                ctype = make_pointer(ctype)
            current_function.argtypes[line[1]] = ctype
        elif line[0] == "deprecated":
            current_function.deprecated = line[1]
        elif line[0] == "version":
            current_function.version = line[1]
                
    return functions


outfile = open("rpigl/gles2.py", "w")

gles2_header = """
import ctypes
from .load_gles import load_gl_proc, GLError

"""

to_be_deleted = ["ctypes", "load_gl_proc"]

outfile.write(gles2_header)

gl_spec = open("specfiles/gl.spec.FIXED", "r")
types = set()
for f in parse_gl_spec(gl_spec):
    if f.valid and not f.deprecated and not f.extension and has_function(f.name):
        f.write_loading(outfile)

for p in parse_enums(open("specfiles/enum.spec", "r")):
     outfile.write("GL_%s = %s\n" % p)

for name in to_be_deleted:
     outfile.write("del %s\n" % name)
