"""Create 4x4 transformation matrices."""

import numpy
from numpy import linalg
from numpy.linalg import inv
import math

identity = numpy.asfortranarray(numpy.eye(4, dtype=numpy.float32))

def stretching(sx, sy, sz):
    """Create a transformation matrix that represents a stretching along x, y and z direction."""
    result = numpy.copy(identity)
    result[0, 0] = sx
    result[1, 1] = sy
    result[2, 2] = sz
    return result

def scaling(s):
    """Create a transformation matrix that represents a uniform scaling."""
    return stretching(s, s, s)

def translation(*v):
    """Create a transformation matrix that represents a translation."""
    v = numpy.asarray(v).flatten()
    result = numpy.copy(identity)
    result[:3,3] = v
    return result
    
_basic_ortho2d = numpy.array(
    [[1,  0,  0, -1],
     [0, -1,  0,  1],
     [0,  0, -1,  0],
     [0,  0,  0,  1]], dtype=numpy.float32, order='F')

def ortho2d(width, height):
    """Create a transformation matrix that maps [0,width]x[0,height] into [-1,1]x[-1,1].
    Useful for 2D games that want to see a conventional framebuffer coordinate system."""
    result = numpy.copy(_basic_ortho2d)
    result[0, 0] = 2.0/width
    result[1, 1] = -2.0/height
    return result

_axes = {
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
    """Create a transformation matrix that represents a rotation (in radians) along the given axis."""
    c = math.cos(angle)
    s = math.sin(angle)
    i, j = _axes[axis]
    result = numpy.copy(identity)
    result[i,i] = c
    result[i,j] = -s
    result[j,i] = s
    result[j,j] = c

    return result

def rotation_degrees(degrees, axis="z"):
    """Create a transformation matrix that represents a rotation (in degrees) along the given axis."""
    return rotation(degrees * (math.pi / 180), axis)


def compose(*matrices):
    """Multiply the given transformation matrices together."""
    if len(matrices) == 0:
        return identity
    else:
        m = matrices[0]
        for matrix in matrices[1:]:
            m = numpy.dot(m, matrix)
        return m


def normal_transform(matrix):
    """Compute the 3x3 matrix which transforms normals given an affine vector transform."""
    return inv(numpy.transpose(matrix[:3,:3]))
