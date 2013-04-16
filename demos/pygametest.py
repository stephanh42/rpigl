import pygame
from rpigl import glesutils, transforms
from rpigl.gles2 import *

# A GLSL (GL Shading Language)  program consists of at least two shaders:
# a vertex shader and a fragment shader.
# Here is the vertex shader.
vertex_glsl = """
uniform mat4 mvp_mat; // a uniform is an input to the shader which is the same for all vertices

attribute vec2 vertex_attrib; // an attribute is a vertex-specific input to the vertex shader
attribute vec2 texcoord_attrib; // an attribute is a vertex-specific input to the vertex shader

varying vec2 texcoord_var;  // a varying is output to the vertex shader and input to the fragment shader

void main(void) {
  gl_Position = mvp_mat * vec4(vertex_attrib, 0.0, 1.0);
  texcoord_var = texcoord_attrib;
}
"""

# Here is the fragment shader
fragment_glsl = """
uniform sampler2D texture; // access the texture
varying vec2 texcoord_var;
void main(void) {
  gl_FragColor = texture2D(texture, texcoord_var);
}
"""

# The array spec: names and formats of the per-vertex attributes
#   vertex_attrib:2h  = two signed short integers  
#   color_attrib:3Bn  = three unsigned bytes, normalized (i.e. shader scales number 0..255 back to a float in range 0..1)
array_spec = glesutils.ArraySpec("vertex_attrib,texcoord_attrib:2h")

class MyWindow(glesutils.GameWindow):
    framerate = 20
    angle = 0

    def init(self):
        """All setup which requires the OpenGL context to be active."""
        
        # compile vertex and fragment shaders
        vertex_shader = glesutils.VertexShader(vertex_glsl)
        fragment_shader = glesutils.FragmentShader(fragment_glsl)
        # link them together into a program
        program = glesutils.Program(vertex_shader, fragment_shader)
        # set the background to RGBA = (1, 0, 0, 1) (solid red)
        glClearColor(1, 0, 0, 1)

        # set up pre-multiplied alpha
        glEnable(GL_BLEND)
        glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA)

        # load uniforms
        self.mvp_mat = program.uniform.mvp_mat
        self.mvp_mat.value = transforms.rotation_degrees(self.angle, "z")

        # bind uniform named "texture" to texture unit 1
        # normally, when using only one texture, 0 would be more logical,
        # but this is just for demo purposes
        program.uniform.texture.value = 1 # bind texture to texture unit 1
   
        # data for the three vertices
        positions = ((0, 0), (0, 1), (1, 1), (1, 0))
        elements = (0, 1, 2, 0, 2, 3)
        # create an array buffer from the spec
        # note: all buffer objects are automatically bound on creation
        self.drawing = array_spec.make_drawing(vertex_attrib=positions, elements=elements)

        texture_data = glesutils.TextureData.from_file("apple.png")
        self.texture = glesutils.Texture.from_data(texture_data)
        print("Texture size: %dx%d" % (texture_data.width, texture_data.height))

        # use the program, bind the texture
        program.use()
        self.texture.bind(program.uniform.texture.value)

        # print some OpenGL implementation information
        version = glesutils.get_version()
        for k in version.__dict__:
            print("%s: %s" % (k, getattr(version, k)))
        

    def on_frame(self, time):
        self.angle = self.angle + time*0.05
        self.redraw()

    def draw(self):
        #update uniform
        self.mvp_mat.value = transforms.rotation_degrees(self.angle, "z")

        # draw triangles
        # program must be used
        self.drawing.draw()


MyWindow(640, 480, pygame.RESIZABLE).run()
