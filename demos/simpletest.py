import pygame
from rpigl import glesutils, transforms
from rpigl.gles2 import *

# A GLSL (GL Shading Language)  program consists of at least two shaders:
# a vertex shader and a fragment shader.
# Here is the vertex shader.
vertex_glsl = """
attribute vec2 vertex_attrib; // an attribute is a vertex-specific input to the vertex shader
void main(void) {
  gl_Position = vec4(vertex_attrib, 0.0, 1.0);
}
"""

# Here is the fragment shader
fragment_glsl = """
uniform vec4 color;
void main(void) {
  gl_FragColor = color;
}
"""

# The array spec: names and formats of the per-vertex attributes
array_spec = glesutils.ArraySpec("vertex_attrib:2f")

class MyWindow(glesutils.GameWindow):

    def init(self):
        """All setup which requires the OpenGL context to be active."""
        
        # compile vertex and fragment shaders
        vertex_shader = glesutils.VertexShader(vertex_glsl)
        fragment_shader = glesutils.FragmentShader(fragment_glsl)
        # link them together into a program
        program1 = glesutils.Program(vertex_shader, fragment_shader)
        program2 = glesutils.Program(vertex_shader, fragment_shader)
        # use the program
        program1.use()
        program1.uniform.color.value = (1, 0, 0, 1)
        program2.use()
        program2.uniform.color.value = (0, 1, 0, 1)
        self.program1 = program1
        self.program2 = program2

        glClearColor(0, 0, 0, 1)

        self.drawing1 = array_spec.make_drawing(vertex_attrib=[(0, 0), (0, 1), (1, 0)])
        self.drawing2 = array_spec.make_drawing(vertex_attrib=[(0, 0), (0, -1), (-1, 0)])

    def draw(self):
        self.program1.use()
        self.drawing1.draw()

        self.program2.use()
        self.drawing2.draw()


MyWindow(640, 480, pygame.RESIZABLE).run()
