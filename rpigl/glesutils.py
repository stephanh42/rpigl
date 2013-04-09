"""
Utility functions that lightly wrap some GL functionality.
Also interaction with pygame.
"""

from . import gles2
from .rpi_egl import create_opengl_window
import ctypes
import pygame
import numpy
import re
from operator import attrgetter
from functools import partial
from .lazycall import LazyAttr

def _get_params(f, *args):
    params = ctypes.c_int32(0)
    f(*args + (params,))
    return params.value


def load_shader(code, shader_type):
    """Load an OpenGL shader given the GLSL source code as a string."""
    shader = gles2.glCreateShader(shader_type)
    code = code.encode()
    gles2.glShaderSource(shader, 1, ctypes.byref(ctypes.c_char_p(code)), None);
    gles2.glCompileShader(shader)
    status = _get_params(gles2.glGetShaderiv, shader, gles2.GL_COMPILE_STATUS)
    if status == gles2.GL_FALSE:
        log_length = _get_params(gles2.glGetShaderiv, shader, gles2.GL_INFO_LOG_LENGTH)
        log = ctypes.create_string_buffer(log_length)
        gles2.glGetShaderInfoLog(shader, log_length, None, log)
        log = log.value
        raise gles2.GLError("compile error : %s" % log)

    return shader


def create_program(*shaders):
    """Load an OpenGL shader program by linking the given shaders."""
    program = gles2.glCreateProgram()
    for shader in shaders:
        gles2.glAttachShader(program, shader)
    gles2.glLinkProgram(program)
    status = _get_params(gles2.glGetProgramiv, program, gles2.GL_LINK_STATUS)
    if status == gles2.GL_FALSE:
        log_length = _get_params(gles2.glGetProgramiv, shader, gles2.GL_INFO_LOG_LENGTH)
        log = ctypes.create_string_buffer(log_length)
        gles2.glGetProgramInfoLog(shader, log_length, None, log)
        log = log.value
        raise GLError("link error : %s" % log)

    return program


class Shader(object):
    """OpenGL shader object."""

    def __init__(self, code):
        """Create a shader object from the GLSL source code."""
        self.shader = load_shader(code, self.shader_type)

    def delete(self):
        """Delete the shader object."""
        gles2.glDeleteShader(self.shader)
        self.shader = 0


class VertexShader(Shader):
    """OpenGL vertex shader object."""
    shader_type = gles2.GL_VERTEX_SHADER

class FragmentShader(Shader):
    """OpenGL fragment shader object."""
    shader_type = gles2.GL_FRAGMENT_SHADER


class Uniform(object):
    """An OpenGL uniform variable."""

    __slots__ = ["program", "name", "uniform"]

    def __init__(self, program, name):
        """Create a uniform given a program and the name."""
        self.program = program
        self.name = name
        self.uniform = gles2.glGetUniformLocation(program, name.encode())

    def load(self, ar):
        """Load data into the uniform. Its program must be in use."""
        assert _used_program == self.program
        load_uniform(self.uniform, ar)

    def load_int(self, n):
        """Load a single integer into the uniform. Its program must be in use."""
        assert _used_program == self.program
        gles2.glUniform1i(self.uniform, n)

    def __repr__(self):
        return "Uniform(%d, %s)" % (self.program, repr(self.name))


class Attrib(object):
    """An OpenGL per-vertex attribute."""

    __slots__ = ["program", "name", "location", "enabled"]

    def __init__(self, program, name):
        """Create a attrib given a program and the name."""
        self.program = program
        self.name = name
        self.location = gles2.glGetAttribLocation(program, name.encode())
        self.enabled = False

    def enable(self):
        """Enable the attrib. Its program must be in use."""
        assert _used_program == self.program
        gles2.glEnableVertexAttribArray(self.location)
        self.enabled = True

    def disable(self):
        """Disable the attrib. Its program must be in use."""
        assert _used_program == self.program
        gles2.glDisableVertexAttribArray(self.location)
        self.enabled = False

    def __repr__(self):
        return "Attrib(%d, %s)" % (self.program, repr(self.name))


_used_program = None

class Program(object):
    """An OpenGL shader program."""

    __slots__ = ["program", "uniform", "attrib"]

    def __init__(self, *shaders):
        """Link shaders together into a single program."""
        program = create_program(*[shader.shader for shader in shaders])
        self.program = program
        self.uniform = LazyAttr(partial(Uniform, program))
        self.attrib = LazyAttr(partial(Attrib, program))

    def use(self):
        """Start using this program."""
        global _used_program
        gles2.glUseProgram(self.program)
        _used_program = self.program

    def is_used(self):
        """Check if program is currently in use."""
        return _used_program == self.program

    def delete(self):
        """Delete program."""
        gles2.glDeleteProgram(self.program)
        self.program = 0

    def enabled_attribs(self):
        """Return a list of all enabled attribs."""
        return [attrib for attrib in self.attrib if attrib.enabled]

    def disable_all_attribs(self):
        """Disable all enabled attribs."""
        for attrib in self.enabled_attribs():
            attrib.disable()


class AttribSpec(object):
    """The specification for a single attrib."""

    gl_types = {
      'b' : gles2.GL_BYTE,
      'B' : gles2.GL_UNSIGNED_BYTE,
      'h' : gles2.GL_SHORT,
      'H' : gles2.GL_UNSIGNED_SHORT,
      'l' : gles2.GL_INT,
      'L' : gles2.GL_UNSIGNED_INT,
      'f' : gles2.GL_FLOAT,
      'd' : gles2.GL_DOUBLE
    }

    regex = re.compile(r"^(.+):([1-4])([bBhHlLfd])([n]*)$")

    normalized = gles2.GL_FALSE
    offset = 0
    allowed_shapes = frozenset([(), (1,)])

    def __init__(self, spec):
        """Create the AttribSpec from its string representation.

        Format: name[,name]:<count><type><flags>
        where:
          count = 1, 2, 3 or 4
          type  = b, B, h, H, l, L, f or d
          flags = nothing or n (normalized)

        Ex: position:3f
            color:4Bn
        """
        self.spec = spec
        names,count, type, flags = self.regex.match(spec).groups()

        self.names = names.split(",")
        self.count = int(count)
        if self.count != 1:
            self.allowed_shapes = frozenset([(self.count,)])
        self.gl_type = self.gl_types[type]
        self.dtype = numpy.dtype(type)
        self.itemsize = self.dtype.itemsize * self.count
        self.alignment = self.dtype.alignment

        for flag in flags:
            if flag == 'n':
                self.normalized = gles2.GL_TRUE

    def locations(self, program):
        """Get a list of locations of the attribs in the given program."""
        program_attrib = program.attrib
        return [getattr(program_attrib, name).location for name in self.names]

    def prep_array(self, ar):
        ar = numpy.ascontiguousarray(ar, dtype=self.dtype)
        if ar.shape[1:] not in self.allowed_shapes:
            raise ValueError("Invalid array shape: %s" % ar.shape)
        return ar

    def load_array(self, prepped_array, length, offset):
        array_length = min(length - offset, len(prepped_array))
        byte_offset = length * self.offset + offset * self.itemsize
        gles2.glBufferSubData(gles2.GL_ARRAY_BUFFER, byte_offset, array_length * self.itemsize, prepped_array.ctypes.data)

    def __repr__(self):
        return "AttribSpec(%s)" % repr(self.spec)
        


class ArraySpec(object):
    """The specification for an ArrayBuffer format."""

    def __init__(self, spec):
        """Create the ArraySpec from its string representation, which consists of whitespace-separated AttribSpecs."""
        self.spec = spec
        attribs = [AttribSpec(field) for field in spec.split()]
        attribs.sort(key=attrgetter("alignment"), reverse=True)
        self.attribs = attribs

        attrib_dict = {}
        size = 0
        for attrib in attribs:
            attrib.offset = size
            size = size + attrib.itemsize
            for name in attrib.names:
                attrib_dict[name] = attrib

        self.attrib_dict = attrib_dict
        self.size = size

    def create_buffer(self, length):
        """Create an ArrayBuffer of the given length."""
        return ArrayBuffer(self, length)

    def __repr__(self):
        return "ArraySpec(%s)" % repr(self.spec)



class BufferObject(object):
    """Base class for OpenGL buffer objects."""
    bound_buffers = {}
    byte_size = 0

    def __init__(self, usage):
        self.usage = usage
        vbo = ctypes.c_uint(0)
        gles2.glGenBuffers(1, ctypes.byref(vbo))
        self.vbo = vbo

    def bind(self):
        """Bind the buffer object."""
        gles2.glBindBuffer(self.target, self.vbo)
        self.bound_buffers[self.target] = self.vbo

    def is_bound(self):
        return self.bound_buffers.get(self.target, None) == self.vbo

    def delete(self):
        """Delete the buffer object."""
        gles2.glDeleteBuffers(1, ctypes.byref(self.vbo))
        self.vbo.value = 0

    def buffer_data(self, size, ptr=None):
        assert self.is_bound()
        gles2.glBufferData(self.target, size, ptr, self.usage)
        self.byte_size = size

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        """item should be a slice"""
        start, stop, step = item.indices(self.length)
        assert step == 1
        return SlicedBuffer(self, start, stop)


class SlicedBuffer(object):
    __slots__ = ["buffer_object", "start", "stop"]

    def __init__(self, buffer_object, start, stop):
        self.buffer_object = buffer_object
        self.start = start
        self.stop = stop

    def draw(self, mode=gles2.GL_TRIANGLES):
        self.buffer_object.draw(mode, self.start, self.stop - self.start)

    def __len__(self):
        return self.stop - self.start

    def __getitem__(self, item):
        """item should be a slice"""
        start, stop, step = item.indices(self.stop - self.start)
        assert step == 1
        return SlicedBuffer(self.buffer_object, self.start + start, self.start + stop)


class ArrayBuffer(BufferObject):
    """An OpenGL array buffer object."""
    target = gles2.GL_ARRAY_BUFFER

    def __init__(self, spec, length, usage=gles2.GL_STATIC_DRAW):
        """Create an array buffer object from the given ArraySpec and with the given length."""
        BufferObject.__init__(self, usage)
        if not isinstance(spec, ArraySpec):
            spec = ArraySpec(spec)
        self.spec = spec
        self.length = length

        self.bind()
        self.buffer_data(spec.size * length)


    def load(self, attrib, array, offset=0):
        assert self.is_bound()
        attrib = self.spec.attrib_dict[attrib]
        array = attrib.prep_array(array)
        attrib.load_array(array, self.length, offset)


    def draw(self, mode=gles2.GL_TRIANGLES, first=0, count=None):
        assert self.is_bound()
        if count is None:
            count = self.length - first
        gles2.glDrawArrays(mode, first, count)

    def attach(self, program):
        assert self.is_bound()
        assert program.is_used()
        length = self.length
        glVertexAttribPointer = gles2.glVertexAttribPointer
        for attrib_spec in self.spec.attribs:
            count = attrib_spec.count
            gl_type = attrib_spec.gl_type
            normalized = attrib_spec.normalized
            byte_offset = length * attrib_spec.offset
            for location in attrib_spec.locations(program):
                glVertexAttribPointer(location, count, gl_type, normalized, 0, byte_offset)
            

class ElementBuffer(BufferObject):
    """An OpenGL element buffer object."""
    target = gles2.GL_ELEMENT_ARRAY_BUFFER
    gl_types = {'B' : gles2.GL_UNSIGNED_BYTE, 'H' : gles2.GL_UNSIGNED_SHORT}

    def __init__(self, array=None, type='H', usage=gles2.GL_STATIC_DRAW):
        BufferObject.__init__(self, usage)
        self.gl_type = self.gl_types[type]
        self.dtype = numpy.dtype(type)
        self.itemsize = self.dtype.itemsize
        self.bind()
        if array is not None:
            self.load(array)

    def load(self, array):
        assert self.is_bound()
        array = numpy.ravel(numpy.ascontiguousarray(array, dtype=self.dtype))
        self.length = len(array)
        self.buffer_data(self.itemsize * self.length, array.ctypes.data)

    def draw(self, mode=gles2.GL_TRIANGLES, first=0, count=None):
        assert self.is_bound()
        if count is None:
            count = self.length - first
        gles2.glDrawElements(mode, count, self.gl_type, first * self.itemsize)


_uniform_loaders = {
  (1,) : lambda uni, ptr: gles2.glUniform1fv(uni, 1, ptr),
  (2,) : lambda uni, ptr: gles2.glUniform2fv(uni, 1, ptr),
  (3,) : lambda uni, ptr: gles2.glUniform3fv(uni, 1, ptr),
  (4,) : lambda uni, ptr: gles2.glUniform4fv(uni, 1, ptr),
  (2, 2) : lambda uni, ptr: gles2.glUniformMatrix2fv(uni, 1, gles2.GL_FALSE, ptr),
  (3, 3) : lambda uni, ptr: gles2.glUniformMatrix3fv(uni, 1, gles2.GL_FALSE, ptr),
  (4, 4) : lambda uni, ptr: gles2.glUniformMatrix4fv(uni, 1, gles2.GL_FALSE, ptr)
}

c_float_p = ctypes.POINTER(ctypes.c_float)

def load_uniform(uniform, ar):
    ar = numpy.asfortranarray(ar, dtype=numpy.float32)
    loader = _uniform_loaders[ar.shape]
    ptr = ar.ctypes.data_as(c_float_p)
    loader(uniform, ptr)

def _get_array_from_alpha_surface(surface):
    rgb = pygame.surfarray.pixels3d(surface).astype(numpy.uint16)
    alpha = pygame.surfarray.pixels_alpha(surface)
    rgb *= alpha[:,:,numpy.newaxis]
    rgb /= 255
    result = numpy.empty(rgb.shape[:-1] + (4,), dtype=numpy.uint8)
    result[:,:,:3] = rgb
    result[:,:,3] = alpha
    return result


_numpy_formats = {3 : gles2.GL_RGB, 4 : gles2.GL_RGBA}

class TextureData(object):
    """A representation of texture data in main memory, before being uploaded to the GPU."""

    __slots__ = ["pointer", "width", "height", "format", "extra_data"]
    
    def __init__(self, pointer, width, height, format, extra_data=None):
        """Create a TextureData object.
        pointer        -- ctypes pointer to data
        width,height   -- image size
        format         -- an OpenGL format such as GL_RGB or GL_RGBA
        extra_data     -- an arbitrary object which must be kept alive to keep 'pointer' valid
        """
        self.pointer = pointer
        self.width = width
        self.height = height
        self.format = format
        self.extra_data = extra_data

    @staticmethod
    def from_ndarray(ar):
        """Create texture data from a Numpy ndarray."""
        ar = numpy.ascontiguousarray(ar, dtype=numpy.uint8)
        height, width, depth = ar.shape
        format = _numpy_formats[depth]
        return TextureData(ar.ctypes.data, width, height, format, ar)
 
    @staticmethod
    def from_surface(surface):
        """Create texture data from a Pygame Surface."""
        if surface.get_flags() & pygame.SRCALPHA:
            desired_depth = 32
            get_array = _get_array_from_alpha_surface
            convert = surface.convert_alpha
        else:
            desired_depth = 24
            get_array = pygame.surfarray.pixels3d
            convert = surface.convert

        if surface.get_bitsize() != desired_depth:
            surface = convert(desired_depth, surface.get_flags())

        ar = numpy.swapaxes(get_array(surface), 0, 1)
        return TextureData.from_ndarray(ar)

    @staticmethod
    def from_file(filename_or_obj):
        """Create texture data from a file, either indicated by name or by file object."""
        return TextureData.from_surface(pygame.image.load(filename_or_obj))


class CubeFormat(object):
    """Represents a way to lay out the 6 faces of the cube in a single image.
    Example format:
    .Y..
    xzXZ
    .y..
    """

    _side_map = {'x' : 0, 'X' : 1, 'y' : 2, 'Y' : 3, 'z' : 4, 'Z' : 5}

    def __init__(self, format):
        format = format.split()
        height = len(format)
        width = max(len(line) for line in format)

        cube_coords = [None] * 6
        for iy in range(height):
            line = format[iy]
            for ix in range(len(line)):
                ch = line[ix]
                if ch in self._side_map:
                    n = self._side_map[ch]
                    cube_coords[n] = (ix, iy)

        for coord in cube_coords:
            assert coord is not None

        self.format = "\n".join(format)
        self.width = width
        self.height = height
        self.cube_coords = tuple(cube_coords)

    def split(self, surface):
        """Split a single image into 6 images, corresponding to the 6 sides of a cube."""
        w = surface.get_width() // self.width
        h = surface.get_height() // self.height
        cube_coords = self.cube_coords
        return [surface.subsurface(pygame.Rect(w*x, h*y, w, h)) for (x, y) in cube_coords]


split_cube_map = CubeFormat(".y.. XzxZ .Y..").split


_cubemap_targets = (gles2.GL_TEXTURE_CUBE_MAP_POSITIVE_X, gles2.GL_TEXTURE_CUBE_MAP_NEGATIVE_X, 
   gles2.GL_TEXTURE_CUBE_MAP_POSITIVE_Y, gles2.GL_TEXTURE_CUBE_MAP_NEGATIVE_Y, 
   gles2.GL_TEXTURE_CUBE_MAP_POSITIVE_Z, gles2.GL_TEXTURE_CUBE_MAP_NEGATIVE_Z)

_repeats = {
  "none" : (gles2.GL_CLAMP_TO_EDGE, gles2.GL_CLAMP_TO_EDGE),
  "x" : (gles2.GL_REPEAT, gles2.GL_CLAMP_TO_EDGE),
  "y" : (gles2.GL_CLAMP_TO_EDGE, gles2.GL_REPEAT),
  "both" : (gles2.GL_REPEAT, gles2.GL_REPEAT)
  }

class Texture(object):
    """An OpenGL texture object."""

    __slots__ = ["texture", "target"]

    def __init__(self):
        """Create a new texture with initially no texture data."""
        texture = ctypes.c_uint(0)
        gles2.glGenTextures(1, ctypes.byref(texture))
        self.texture = texture

    def bind(self):
        """Bind the texture."""
        gles2.glBindTexture(self.target, self.texture)

    def delete(self):
        """Delete the texture."""
        gles2.glDeleteTextures(1, ctype.byref(self.texture))
        self.texture.value = 0

    def load(self, texture_data, repeat="none", mipmap=True, cubemap=False, conversion=lambda x:x):
        """Load texture data from the given pointer."""
        repeat_s, repeat_t = _repeats[repeat]
        version = get_version()
        mipmap_generation = version.mipmap_generation if mipmap else None

        if cubemap:
            self.target = gles2.GL_TEXTURE_CUBE_MAP
            targets = _cubemap_targets
        else:
            self.target = gles2.GL_TEXTURE_2D
            targets = (self.target,)
            texture_data = (texture_data,)

        assert len(targets) == len(texture_data)

        self.bind()

        gles2.glTexParameteri(self.target, gles2.GL_TEXTURE_MIN_FILTER, gles2.GL_LINEAR_MIPMAP_LINEAR if mipmap_generation else gles2.GL_LINEAR)
        gles2.glTexParameteri(self.target, gles2.GL_TEXTURE_MAG_FILTER, gles2.GL_LINEAR)
        gles2.glTexParameteri(self.target, gles2.GL_TEXTURE_WRAP_S, repeat_s)
        gles2.glTexParameteri(self.target, gles2.GL_TEXTURE_WRAP_T, repeat_t)
        if cubemap and version.gl_version:
            # Apparently this does not work on GLES?
            gles2.glTexParameteri(self.target, gles2.GL_TEXTURE_WRAP_R, gles2.GL_CLAMP_TO_EDGE)

        if mipmap_generation == "GL_GENERATE_MIPMAP":
            gles2.glTexParameteri(self.target, gles2.GL_GENERATE_MIPMAP, gles2.GL_TRUE)

        for i in range(len(targets)):
            data = conversion(texture_data[i])
            gles2.glTexImage2D(targets[i], 0, data.format, data.width, data.height, 0, data.format, gles2.GL_UNSIGNED_BYTE, data.pointer)

        if mipmap_generation == "glGenerateMipmap":
            gles2.glGenerateMipmap(self.target)


    @classmethod
    def from_data(cls, texture_data, **kws):
        """Create texture from texture data."""
        texture = cls()
        texture.load(texture_data, **kws)
        return texture
    
    @classmethod
    def from_ndarray(cls, ar, **kws):
        """Create texture from a Numpy ndarray."""
        return cls.from_data(ar, conversion=TextureData.from_ndarray, **kws)

    @classmethod
    def from_surface(cls, surface, **kws):
        """Create texture from a Pygame Surface."""
        return cls.from_data(surface, conversion=TextureData.from_surface, **kws)

    @classmethod
    def from_file(cls, filename_or_obj, **kws):
        """Create texture from a file, either indictaed by name or by file object."""
        return cls.from_data(filename_or_obj, conversion=TextureData.from_file, **kws)


class Version(object):
    """Consult OpenGL version information"""

    _gl_version_re = re.compile(r"([0-9.]+)\s")
    _gles_version_re = re.compile(r"OpenGL\s+ES(?:-([A-Z]+))?\s+([0-9.]+)")

    def __init__(self):
        self.vendor = gles2.glGetString(gles2.GL_VENDOR).decode()
        self.renderer = gles2.glGetString(gles2.GL_RENDERER).decode()
        self.version = gles2.glGetString(gles2.GL_VERSION).decode()
        self.extensions = frozenset(ext.decode() for ext in gles2.glGetString(gles2.GL_EXTENSIONS).split())

        mo = self._gl_version_re.match(self.version)
        if mo is not None:
            self.gl_version = tuple(int(n) for n in mo.group(1).split("."))
        else:
            self.gl_version = ()

        mo = self._gles_version_re.match(self.version)
        if mo is not None:
            gles_version = tuple(int(n) for n in mo.group(2).split("."))
            self.gles_version = gles_version
            self.gles_profile = mo.group(1)
        else:
            self.gles_version = ()
            self.gles_profile = ""

        if self.has_gl_version(3) or self.has_gles_version(2) or self.has_extension("GL_ARB_framebuffer_object"):
            mipmap_generation = "glGenerateMipmap"
        elif self.has_gl_version(1, 4):
            mipmap_generation = "GL_GENERATE_MIPMAP"
        else:
            mipmap_generation = None
        self.mipmap_generation = mipmap_generation


    def has_gl_version(self, *desired_version):
        return desired_version <= self.gl_version

    def has_gles_version(self, *desired_version):
        return desired_version <= self.gles_version

    def has_extension(self, extension):
        return extension in self.extensions



_version = None

def get_version():
    """Get a cached Version object."""
    global _version
    if _version is None:
        _version = Version()
    return _version


class EventHandler(object):
    """Base class which dispatches a Pygame event to an event handler based on event type."""

    callbacks = {
      pygame.QUIT: attrgetter("on_quit"),
      pygame.ACTIVEEVENT: attrgetter("on_activeevent"),
      pygame.KEYDOWN: attrgetter("on_keydown"),
      pygame.KEYUP: attrgetter("on_keyup"),
      pygame.MOUSEMOTION: attrgetter("on_mousemotion"),
      pygame.MOUSEBUTTONUP: attrgetter("on_mousebuttonup"),
      pygame.MOUSEBUTTONDOWN: attrgetter("on_mousebuttondown"),
      pygame.JOYAXISMOTION: attrgetter("on_joyaxismotion"),
      pygame.JOYBALLMOTION: attrgetter("on_joyballmotion"),
      pygame.JOYHATMOTION: attrgetter("on_joyhatmotion"),
      pygame.JOYBUTTONUP: attrgetter("on_joybuttonup"),
      pygame.JOYBUTTONDOWN: attrgetter("on_joybuttondown"),
      pygame.VIDEORESIZE: attrgetter("on_videoresize"),
      pygame.VIDEOEXPOSE: attrgetter("on_videoexpose"),
      pygame.USEREVENT: attrgetter("on_userevent")
    }
    default_callback = attrgetter("on_unknown_event")

    def on_quit(self, event): pass
    def on_activeevent(self, event): pass
    def on_keydown(self, event): pass
    def on_keyup(self, event): pass
    def on_mousemotion(self, event): pass
    def on_mousebuttonup(self, event): pass
    def on_mousebuttondown(self, event): pass
    def on_joyaxismotion(self, event): pass
    def on_joyballmotion(self, event): pass
    def on_joyhatmotion(self, event): pass
    def on_joybuttonup(self, event): pass
    def on_joybuttondown(self, event): pass
    def on_videoresize(self, event): pass
    def on_videoexpose(self, event): pass
    def on_userevent(self, event): pass

    def on_unknown_event(self, event): pass

    def on_event(self, event):
        """Handle an event by dispatching to the appropriate handler."""
        callback = self.callbacks.get(event.type, self.default_callback)
        (callback(self))(event)



class GameWindow(EventHandler):
    done = False
    clear_flags = gles2.GL_COLOR_BUFFER_BIT|gles2.GL_DEPTH_BUFFER_BIT

    clock = pygame.time.Clock()
    framerate = 0
    on_frame_called = True
    
    def __init__(self, width, height, flags=0):
        window = create_opengl_window(width, height, flags)
        self.swap_buffers = window.swap_buffers
        self.window = window
        self.width = window.width
        self.height = window.height

    def init(self):
        pass

    def draw(self):
        pass

    def redraw(self):
        self.init()
        self.init = lambda: None

        gles2.glViewport(0, 0, self.width, self.height)
        gles2.glClear(self.clear_flags)
        self.draw()
        self.swap_buffers()

    def on_quit(self, event):
        self.done = True

    def on_videoresize(self, event):
        window = self.window
        window.on_resize(event.w, event.h)
        self.width = window.width
        self.height = window.height
        self.redraw()

    def on_videoexpose(self, event):
        self.redraw()

    def on_keydown(self, event):
        if event.key == pygame.K_ESCAPE:
            self.done = True

    def on_frame(self, time):
        self.redraw()

    def do_one_event(self):
        framerate = self.framerate
        if framerate > 0:
            event = pygame.event.poll()
            if event.type != pygame.NOEVENT:
                self.on_event(event)
            elif self.on_frame_called:
                self.clock.tick(framerate)
                self.on_frame_called = False
            else:
                self.on_frame(self.clock.get_time())
                self.on_frame_called = True
        else:
            self.on_event(pygame.event.wait())


    def run(self):
        pygame.init()
        self.done = False
        try:
            self.redraw()
            self.clock.tick()
            while not self.done:
                self.do_one_event()
        finally:
            pygame.quit()
