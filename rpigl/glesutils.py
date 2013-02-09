"""

Utility functions that lightly wrap some GL functionality.
Also interaction with pygame.
"""

import gles2
from rpi_egl import create_opengl_window
import ctypes
import pygame
import numpy
import re
import string
from operator import attrgetter
from functools import partial
from lazycall import LazyAttr

def _get_params(f, *args):
    params = ctypes.c_int32(0)
    f(*args + (params,))
    return params.value


def load_shader(code, shader_type):
    """Load an OpenGL shader given the GLSL source code as a string."""
    shader = gles2.glCreateShader(shader_type)
    gles2.glShaderSource(shader, 1, ctypes.byref(ctypes.c_char_p(code)), None);
    gles2.glCompileShader(shader)
    status = _get_params(gles2.glGetShaderiv, shader, gles2.GL_COMPILE_STATUS)
    if status == gles2.GL_FALSE:
        log_length = _get_params(gles2.glGetShaderiv, shader, gles2.GL_INFO_LOG_LENGTH)
        log = ctypes.create_string_buffer(log_length)
        gles2.glGetShaderInfoLog(shader, log_length, None, log)
        log = log.value
        raise gles2.GLError, "compile error : %s" % log

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
        raise GLError, "link error : %s" % log

    return program


class Shader:
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


class Uniform:
    """An OpenGL uniform variable."""

    def __init__(self, program, name):
        """Create a uniform given a program and the name."""
        self.program = program
        self.name = name
        self.uniform = gles2.glGetUniformLocation(program, name)

    def load(self, ar):
        """Load data into the uniform. Its program must be in use."""
        assert _used_program == self.program
        load_uniform(self.uniform, ar)

    def __repr__(self):
        return "Uniform(%d, %s)" % (self.program, repr(self.name))


class Attrib:
    """An OpenGL per-vertex attribute."""

    enabled = False

    def __init__(self, program, name):
        """Create a attrib given a program and the name."""
        self.program = program
        self.name = name
        self.location = gles2.glGetAttribLocation(program, name)

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

class Program:
     """An OpenGL shader program."""

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


class AttribSpec:
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

        self.names = string.split(names, ",")
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
            raise ValueError, "Invalid array shape: %s" % ar.shape
        return ar

    def load_array(self, prepped_array, length, offset):
        array_length = min(length - offset, len(prepped_array))
        byte_offset = length * self.offset + offset * self.itemsize
        gles2.glBufferSubData(gles2.GL_ARRAY_BUFFER, byte_offset, array_length * self.itemsize, prepped_array.ctypes.data)

    def __repr__(self):
        return "AttribSpec(%s)" % repr(self.spec)
        


class ArraySpec:

    def __init__(self, spec):
        self.spec = spec
        attribs = [AttribSpec(field) for field in string.split(spec)]
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
        return ArrayBuffer(self, length)

    def __repr__(self):
        return "ArraySpec(%s)" % repr(self.spec)



class BufferObject:
    bound_buffers = {}

    def __init__(self, usage):
        self.usage = usage
        vbo = ctypes.c_uint(0)
        gles2.glGenBuffers(1, ctypes.byref(vbo))
        self.vbo = vbo

    def bind(self):
        gles2.glBindBuffer(self.target, self.vbo)
        self.bound_buffers[self.target] = self.vbo

    def is_bound(self):
        return self.bound_buffers.get(self.target, None) == self.vbo

    def delete(self):
        gles2.glDeleteBuffers(1, ctypes.byref(self.vbo))
        self.vbo.value = 0

    def buffer_data(self, size, ptr=None):
        assert self.is_bound()
        gles2.glBufferData(self.target, size, ptr, self.usage)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        """item should be a slice"""
        start, stop, step = item.indices(self.length)
        assert step == 1
        return SlicedBuffer(self, start, stop)


class SlicedBuffer:
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
    target = gles2.GL_ARRAY_BUFFER

    def __init__(self, spec, length, usage=gles2.GL_STATIC_DRAW):
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


class EventHandler:
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
