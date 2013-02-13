import pygame
import numpy
from math import sqrt
from rpigl import glesutils, transforms
from rpigl.gles2 import *

a = sqrt(2.0/(5.0 + sqrt(5.0)));
b = sqrt(2.0/(5.0 - sqrt(5.0)));

vertices = numpy.array((
    (-a, 0.0, b), (a, 0.0, b), (-a, 0.0, -b), (a, 0.0, -b),
    (0.0, b, a), (0.0, b, -a), (0.0, -b, a), (0.0, -b, -a),
    (b, a, 0.0), (-b, a, 0.0), (b, -a, 0.0), (-b, -a, 0.0)
  ))

indices = numpy.array((
    (1, 4, 0), (4, 9, 0), (4, 5, 9), (8, 5, 4), (1, 8, 4),
    (1, 10, 8), (10, 3, 8), (8, 3, 5), (3, 2, 5), (3, 7, 2),
    (3, 10, 7), (10, 6, 7), (6, 11, 7), (6, 0, 11), (6, 1, 0),
    (10, 1, 6), (11, 0, 9), (2, 11, 9), (5, 2, 9), (11, 2, 7)
  ), dtype=numpy.uint16)

def combine_indices(p0, p1):
    result = numpy.empty((len(p0), 2), dtype=p0.dtype)
    numpy.minimum(p0, p1, result[:,0])
    numpy.maximum(p0, p1, result[:,1])
    return [tuple(p) for p in result]


def refine(vertices, indices):
    # Note: the del statements here try to free potentially
    # large objects as soon as possible, in order to reduce the
    # total  memory footprint and perhaps make cache behaviour a bit nicer.
    dtype = indices.dtype

    p0 = indices[:,0]
    p1 = indices[:,1]
    p2 = indices[:,2]
    del indices
    p01 = combine_indices(p0, p1)
    p12 = combine_indices(p1, p2)
    p02 = combine_indices(p0, p2)

    new_indices = list(set(p01) | set(p02) | set(p12))
    N = len(vertices)
    new_indices_map = {new_indices[i] : i + N for i in range(len(new_indices))} 
    new_indices = numpy.array(new_indices, dtype=dtype)

    new_vertices = vertices[new_indices[:,0],:] + vertices[new_indices[:,1],:]
    new_indices = None # del new_indices doesn't work in Python 2.7

    norms = numpy.add.reduce(new_vertices * new_vertices, 1)
    numpy.sqrt(norms, norms)
    new_vertices /= norms[:,numpy.newaxis]
    del norms

    p01 = numpy.array([new_indices_map[p] for p in p01], dtype=dtype)
    p02 = numpy.array([new_indices_map[p] for p in p02], dtype=dtype)
    p12 = numpy.array([new_indices_map[p] for p in p12], dtype=dtype)
    del new_indices_map

    indices = numpy.vstack([numpy.column_stack(triple) for triple in ((p0, p01, p02), (p01, p1, p12), (p01, p12, p02), (p12, p2, p02))])
    vertices = numpy.vstack((vertices, new_vertices))
    return vertices, indices


for i in range(3):
    vertices, indices = refine(vertices, indices)

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
        glEnable(GL_CULL_FACE)
        glFrontFace(GL_CW)

        glClearColor(1, 0, 0, 1)

#        program.uniform.mvp_mat.load(transforms.rotation_degrees(45, "z"))
        program.uniform.mvp_mat.load(transforms.identity)
        program.uniform.light_dir.load((0, 1, -1))
        program.attrib.vertex_attrib.enable()
        program.attrib.normal_attrib.enable()
        print(program.enabled_attribs())

        vbo = array_spec.create_buffer(len(vertices))
        vbo.load("vertex_attrib", vertices)
        vbo.attach(program)
        self.vbo = vbo

        self.elem_vbo = glesutils.ElementBuffer(indices, type='H')

        print(glGetString(GL_VENDOR))
        print(glGetString(GL_VERSION))
        print(glGetString(GL_EXTENSIONS))

        print("Total VBO size: %gKb" % ((self.vbo.byte_size + self.elem_vbo.byte_size)/1024.0))


    def draw(self):
        self.elem_vbo[:].draw()


MyWindow(640, 640, pygame.RESIZABLE).run()
