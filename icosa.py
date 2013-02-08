import pygame
from math import sqrt
from rpigl import glesutils, transforms
from rpigl.gles2 import *

a = sqrt(2.0/(5.0 + sqrt(5.0)));
b = sqrt(2.0/(5.0 - sqrt(5.0)));

vertices = (
    (-a, 0.0, b), (a, 0.0, b), (-a, 0.0, -b), (a, 0.0, -b),
    (0.0, b, a), (0.0, b, -a), (0.0, -b, a), (0.0, -b, -a),
    (b, a, 0.0), (-b, a, 0.0), (b, -a, 0.0), (-b, -a, 0.0)
  )

indices = (
    (1, 4, 0), (4, 9, 0), (4, 5, 9), (8, 5, 4), (1, 8, 4),
    (1, 10, 8), (10, 3, 8), (8, 3, 5), (3, 2, 5), (3, 7, 2),
    (3, 10, 7), (10, 6, 7), (6, 11, 7), (6, 0, 11), (6, 1, 0),
    (10, 1, 6), (11, 0, 9), (2, 11, 9), (5, 2, 9), (11, 2, 7)
  )


vertex_glsl = """
uniform mat4 mvp_mat;
attribute vec3 vertex_attrib;
attribute vec3 normal_attrib;
varying vec3 normal_var;
void main(void) {
  gl_Position = mvp_mat * vec4(vertex_attrib, 1.0);
  normal_var = normal_attrib;
}
"""

fragment_glsl = """
varying vec3 normal_var;
uniform vec3 light_dir;
void main(void) {
  float s = dot(normal_var, light_dir); 
  gl_FragColor = vec4(s, s, s, 1.0);
}
"""

array_spec = glesutils.ArraySpec("vertex_attrib,normal_attrib:3f")

class MyWindow(glesutils.GameWindow):

    def init(self):
        vertex_shader = glesutils.VertexShader(vertex_glsl)
        fragment_shader = glesutils.FragmentShader(fragment_glsl)
        program = glesutils.Program(vertex_shader, fragment_shader)

        vertex_shader.delete()
        fragment_shader.delete()

        program.use()

        glClearDepthf(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)

        glClearColor(1, 0, 0, 1)

#        program.uniform.mvp_mat.load(transforms.rotation_degrees(45, "z"))
        program.uniform.mvp_mat.load(transforms.identity)
        program.uniform.light_dir.load((0, 1, -1))
        program.attrib.vertex_attrib.enable()
        program.attrib.normal_attrib.enable()
        print program.enabled_attribs()

        vbo = array_spec.create_buffer(len(vertices))
        vbo.load("vertex_attrib", vertices)
        vbo.attach(program)
        self.vbo = vbo

        self.elem_vbo = glesutils.ElementBuffer(indices, type='B')

        print glGetString(GL_VENDOR)
        print glGetString(GL_VERSION)
        print glGetString(GL_EXTENSIONS)


    def draw(self):
        self.elem_vbo[:].draw()


MyWindow(640, 640, pygame.RESIZABLE).run()
