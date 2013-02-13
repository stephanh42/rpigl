"""
Create a window for OpenGL (ES) rendering in a platform-dependent way.

The main entry point is create_opengl_window.
"""


import ctypes
import ctypes.util
import functools
from ctypes import c_int, c_uint, c_int32, c_uint32, c_int16, c_uint16, c_void_p, POINTER
import pygame

# types

c_uint32_p = POINTER(c_uint32)
DISPMANX_DISPLAY_HANDLE_T = c_uint32
DISPMANX_UPDATE_HANDLE_T = c_uint32
DISPMANX_ELEMENT_HANDLE_T = c_uint32
DISPMANX_RESOURCE_HANDLE_T = c_uint32
DISPMANX_PROTECTION_T = c_uint32
DISPMANX_TRANSFORM_T = c_int

class VC_RECT_T(ctypes.Structure):
  _fields_ = [("x", c_int32), ("y", c_int32), ("width", c_int32), ("height", c_int32)]

class VC_DISPMANX_ALPHA_T(ctypes.Structure):
  _fields_ = [("flags", c_int), ("opacity", c_uint32), ("mask", c_void_p)]

class EGL_DISPMANX_WINDOW_T(ctypes.Structure):
  _fields_ = [("element", DISPMANX_ELEMENT_HANDLE_T), ("width", c_int), ("height", c_int)]

EGLint = c_int32
EGLBoolean = c_uint
EGLDisplay = c_void_p
EGLContext = c_void_p
EGLSurface = c_void_p
EGLConfig = c_void_p
EGLNativeDisplayType = c_void_p
EGLNativeWindowType = POINTER(EGL_DISPMANX_WINDOW_T)


# flags

DISPMANX_FLAGS_ALPHA_FIXED_ALL_PIXELS = 1
DISPMANX_PROTECTION_NONE = 0

EGL_CONTEXT_CLIENT_VERSION = 0x3098
EGL_BLUE_SIZE = 0x3022
EGL_GREEN_SIZE = 0x3023
EGL_RED_SIZE = 0x3024
EGL_DEPTH_SIZE = 0x3025
EGL_NONE = 0x3038
EGL_DEFAULT_DISPLAY = None
EGL_NO_CONTEXT = None

# functions

def load(lib, name, restype, *args):
    return (ctypes.CFUNCTYPE(restype, *args))((name, lib))

bcm_host_path = ctypes.util.find_library("bcm_host")
is_raspberry_pi = (bcm_host_path is not None)

if is_raspberry_pi:
    bcm_host_lib = ctypes.CDLL(bcm_host_path)
    gles_lib = ctypes.CDLL(ctypes.util.find_library("GLESv2"))
    egl_lib = ctypes.CDLL(ctypes.util.find_library("EGL"))

    load_bcm_host = functools.partial(load, bcm_host_lib)
    load_egl = functools.partial(load, egl_lib)

    bcm_host_init = load_bcm_host("bcm_host_init", None)
    graphics_get_display_size = load_bcm_host("graphics_get_display_size", c_int32, c_uint16, c_uint32_p, c_uint32_p)
    vc_dispmanx_display_open = load_bcm_host("vc_dispmanx_display_open", DISPMANX_DISPLAY_HANDLE_T, c_uint32)
    vc_dispmanx_update_start = load_bcm_host("vc_dispmanx_update_start", DISPMANX_UPDATE_HANDLE_T, c_int32)

    vc_dispmanx_element_add = load_bcm_host("vc_dispmanx_element_add", DISPMANX_ELEMENT_HANDLE_T,
         DISPMANX_UPDATE_HANDLE_T, DISPMANX_DISPLAY_HANDLE_T,
         c_int32, POINTER(VC_RECT_T), DISPMANX_RESOURCE_HANDLE_T,
         POINTER(VC_RECT_T), DISPMANX_PROTECTION_T,
         POINTER(VC_DISPMANX_ALPHA_T),
         c_void_p, DISPMANX_TRANSFORM_T)

    vc_dispmanx_update_submit_sync = load_bcm_host("vc_dispmanx_update_submit_sync", c_int, DISPMANX_UPDATE_HANDLE_T)

    eglGetDisplay = load_egl("eglGetDisplay", EGLDisplay, EGLNativeDisplayType)
    eglInitialize = load_egl("eglInitialize", EGLBoolean, EGLDisplay , POINTER(EGLint), POINTER(EGLint))
    eglChooseConfig = load_egl("eglChooseConfig", EGLBoolean, EGLDisplay, POINTER(EGLint),
                               POINTER(EGLConfig), EGLint,
                               POINTER(EGLint))

    eglCreateWindowSurface = load_egl("eglCreateWindowSurface", EGLSurface, EGLDisplay, EGLConfig, EGLNativeWindowType, POINTER(EGLint))
    eglCreateContext = load_egl("eglCreateContext", EGLContext, EGLDisplay, EGLConfig, EGLContext, POINTER(EGLint))
    eglMakeCurrent = load_egl("eglMakeCurrent", EGLBoolean, EGLDisplay, EGLSurface, EGLSurface, EGLContext)
    eglSwapBuffers = load_egl("eglSwapBuffers", EGLBoolean, EGLDisplay, EGLSurface)


def make_array(ctype, contents):
    array_type = ctype * len(contents)
    return array_type(*contents)

class BaseWindow:
    def on_resize(self, width, height):
        self.width = width
        self.height = height


class RaspberryWindow(BaseWindow):
    def __init__(self):
       DISPLAY_ID = 0
       
       # create an EGL window surface, passing context width/height
       width = c_uint32(0)
       height = c_uint32(0)
       status = graphics_get_display_size(DISPLAY_ID, ctypes.byref(width), ctypes.byref(height))
       if status < 0:
           raise Exception("cannot obtain display size")
       self.width = width.value
       self.height = height.value

       dst_rect = VC_RECT_T(0, 0, self.width, self.height)
       src_rect = VC_RECT_T(0, 0, self.width << 16, self.height << 16)

       alpha = VC_DISPMANX_ALPHA_T(flags=DISPMANX_FLAGS_ALPHA_FIXED_ALL_PIXELS, opacity=255, mask=None)

       dispman_display = vc_dispmanx_display_open(DISPLAY_ID)
       dispman_update = vc_dispmanx_update_start(0)
             
       dispman_element = vc_dispmanx_element_add(dispman_update, dispman_display,
          0, ctypes.byref(dst_rect), 0,
          ctypes.byref(src_rect), DISPMANX_PROTECTION_NONE, ctypes.byref(alpha), None, 0)
          
       vc_dispmanx_update_submit_sync(dispman_update);

       self.window = EGL_DISPMANX_WINDOW_T(dispman_element, self.width, self.height)


    def create_EGL_context(self, gl_version, attrib_list):
       contextAttribs = make_array(EGLint, [EGL_CONTEXT_CLIENT_VERSION, gl_version, EGL_NONE])
       
       display = eglGetDisplay(EGL_DEFAULT_DISPLAY)
       if not display:
           raise Exception("no EGL display")

       major_version = EGLint(0)
       minor_version = EGLint(0)
       if not eglInitialize(display, ctypes.byref(major_version), ctypes.byref(minor_version)):
         raise Exception("cannot initialize EGL display")

       config = EGLConfig()
       num_configs = EGLint(0)
       if not eglChooseConfig(display, attrib_list, ctypes.byref(config), 1, ctypes.byref(num_configs)):
         raise Exception("no suitable configs available on display")

       surface = eglCreateWindowSurface(display, config, ctypes.byref(self.window), None)
       if not surface:
         raise Exception("cannot create window surface")

       context = eglCreateContext(display, config, EGL_NO_CONTEXT, contextAttribs)
       if not context:
         raise Exception("cannot create GL context")
       
       if not eglMakeCurrent(display, surface, surface, context):
         raise Exception("cannot make GL context current")
       
       self.display = display
       self.surface = surface
       self.context = context


    def swap_buffers(self):
       eglSwapBuffers(self.display, self.surface)


def rpi_create_window(gl_version, attrib_list):
    bcm_host_init();

    window = RaspberryWindow()
    window.create_EGL_context(gl_version, attrib_list)

    return window


class WindowsDesktopWindow(BaseWindow):
    def __init__(self, screen):
        self.width = screen.get_width()
        self.height = screen.get_height()

    def swap_buffers(self):
        pygame.display.flip()


class DesktopWindow(WindowsDesktopWindow):
    def __init__(self, screen, flags):
        WindowsDesktopWindow.__init__(self, screen)
        self.flags = flags

    def on_resize(self, width, height):
        screen = pygame.display.set_mode((width, height), self.flags)
        WindowsDesktopWindow.on_resize(self, screen.get_width(), screen.get_height())


def create_opengl_window(width, height, flags=0):
    """Create a window and establish an OpenGL (ES) context.
    The width and height are merely hints, and ignored on the Pi itself since
    there we always go fullscreen.

    The flags parameter are additional pygame flags; OPENGL and DOUBLEBUF are already implied.
    """
    if is_raspberry_pi:
        attribList = make_array(EGLint, [EGL_RED_SIZE, 8, EGL_GREEN_SIZE, 8, EGL_BLUE_SIZE, 8, EGL_DEPTH_SIZE, 24, EGL_NONE])
        window = rpi_create_window(2, attribList)
        pygame.display.set_mode((window.width, window.height), 0)
    else:
        import platform
        flags = flags|pygame.OPENGL|pygame.DOUBLEBUF
        screen = pygame.display.set_mode((width, height), flags)
        if platform.system() == "Windows":
            window = WindowsDesktopWindow(screen)
        else:
            window = DesktopWindow(screen, flags)

    return window
