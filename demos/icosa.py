import pygame
import numpy
from math import sqrt
from rpigl import glesutils, transforms
from rpigl.gles2 import *

_a = sqrt(2.0/(5.0 + sqrt(5.0)));
_b = sqrt(2.0/(5.0 - sqrt(5.0)));

_vertices = numpy.array((
    (-_a, 0.0, _b), (_a, 0.0, _b), (-_a, 0.0, -_b), (_a, 0.0, -_b),
    (0.0, _b, _a), (0.0, _b, -_a), (0.0, -_b, _a), (0.0, -_b, -_a),
    (_b, _a, 0.0), (-_b, _a, 0.0), (_b, -_a, 0.0), (-_b, -_a, 0.0)
  ))

_indices = numpy.array((
    (1, 4, 0), (4, 9, 0), (4, 5, 9), (8, 5, 4), (1, 8, 4),
    (1, 10, 8), (10, 3, 8), (8, 3, 5), (3, 2, 5), (3, 7, 2),
    (3, 10, 7), (10, 6, 7), (6, 11, 7), (6, 0, 11), (6, 1, 0),
    (10, 1, 6), (11, 0, 9), (2, 11, 9), (5, 2, 9), (11, 2, 7)
  ), dtype=numpy.uint16)

def make_sphere(level):
    """Create an approximation of a sphere using 20 * (4 ** level) triangles.
    Returns a pair of vertex coordinate and element indices.
    """
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

    vertices = _vertices
    indices = _indices
    for i in range(level):
        vertices, indices = refine(vertices, indices)
    return vertices, indices


array_spec = glesutils.ArraySpec("vertex_attrib,normal_attrib,texcoord_attrib:3f")

vertex_glsl = array_spec.glsl() + """
uniform mat4 mvp_mat;
uniform mat3 texcoord_mat;

varying vec3 normal_var;
varying vec3 texcoord_var;

void main(void) {
  gl_Position = mvp_mat * vec4(vertex_attrib, 1.0);
  normal_var = normal_attrib;
  texcoord_var = texcoord_mat * texcoord_attrib;
}
"""

fragment_glsl = """
uniform samplerCube texture;
uniform vec3 light_dir;

varying vec3 normal_var;
varying vec3 texcoord_var;

void main(void) {
  float s = dot(normal_var, light_dir); 
  vec4 color = textureCube(texture, texcoord_var);
  gl_FragColor = s * color;
}
"""

class MyWindow(glesutils.GameWindow):

    angle = 0
    framerate = 20

    def init(self):
        vertices, indices = make_sphere(3)
        print(len(indices))
        self.drawing = array_spec.make_drawing(vertex_attrib=vertices, elements=indices)

        vertex_shader = glesutils.VertexShader(vertex_glsl)
        fragment_shader = glesutils.FragmentShader(fragment_glsl)
        program = glesutils.Program(vertex_shader, fragment_shader)
        self.program = program

        program.use()

        glClearDepthf(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_CULL_FACE)
        glFrontFace(GL_CW)

        glClearColor(1, 0, 0, 1)

#        program.uniform.mvp_mat.load(transforms.rotation_degrees(45, "z"))
        program.uniform.mvp_mat.value = transforms.identity
        program.uniform.light_dir.value = (0, 1, -1)
        program.uniform.texture.value = 0

        img = pygame.image.load("world_cube_net.png")
        self.texture = glesutils.Texture.from_surface(glesutils.split_cube_map(img), cubemap=True)
        self.texture.bind(0)
 
        print(glGetString(GL_VENDOR))
        print(glGetString(GL_VERSION))
        print(glGetString(GL_EXTENSIONS))


    def on_frame(self, time):
        self.angle = self.angle + time*0.05
        self.redraw()

    def draw(self):
        m = transforms.compose(transforms.rotation_degrees(self.angle, "y"), transforms.stretching(-1, 1, 1))
        self.program.uniform.texcoord_mat.value = m[:3,:3]
        self.drawing.draw()


MyWindow(640, 640, pygame.RESIZABLE).run()
