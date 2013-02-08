import numpy
from numpy import linalg
import math

identity = numpy.asfortranarray(numpy.eye(4, dtype=numpy.float32))

def stretching(sx, sy, sz):
    result = numpy.copy(identity)
    result[0, 0] = sx
    result[1, 1] = sy
    result[2, 2] = sz
    return result

def scaling(s):
    return stretching(s, s, s)

def translation(*v):
    v = numpy.asarray(v).flatten()
    result = numpy.copy(identity)
    result[:3,3] = v
    return result
    
basic_ortho2d = numpy.array(
    [[1,  0,  0, -1],
     [0, -1,  0,  1],
     [0,  0, -1,  0],
     [0,  0,  0,  1]], dtype=numpy.float32, order='F')

def ortho2d(width, height):
    result = numpy.copy(basic_ortho2d)
    result[0, 0] = 2.0/width
    result[1, 1] = -2.0/height
    return result

axes = {
  "x": (1, 2),
  "y": (2, 0),
  "z": (0, 1),
  "xy" : (0, 1),
  "yx" : (1, 0),
  "xz" : (0, 2),
  "zx" : (2, 0),
  "yz" : (1, 2),
  "zy" : (2, 1),
}

def rotation(angle, axis="z"):
    c = math.cos(angle)
    s = math.sin(angle)
    i, j = axes[axis]
    result = numpy.copy(identity)
    result[i,i] = c
    result[i,j] = -s
    result[j,i] = s
    result[j,j] = c

    return result

def rotation_degrees(degrees, axis="z"):
    return rotation(degrees * (math.pi / 180), axis)


def compose(*matrices):
    if len(matrices) == 0:
        return identity
    else:
        m = matrices[0]
        for matrix in matrices[1:]:
            m = numpy.dot(m, matrix)
        return m
