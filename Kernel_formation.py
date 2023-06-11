import numpy as np
from PIL import Image
import os


kernel_1 = np.array([[0, 0, 0],
                     [0, 1, 1],
                     [0, 0, 0]])

matrix = np.random.randn(480, 640)

def main():
    convolve = image_convolution(matrix, kernel_1, 480, 640, 2, 2)
    print np.shape(convolve)
    change = image_convolution(matrix, convolve, 480, 640, 477, 637)
    print change




def image_convolution(matrix, kernel, m_height, m_width, a, b):
    # assuming kernel is symmetric and odd
    matrix = matrix.reshape(m_height, m_width)

    output = []
    for i in range(m_height - a):
        print i
        for j in range(m_width - b):
            output.append(np.sum(matrix[i:a + i, j:b + j] * kernel))

    output = np.array(output)

    return output.reshape((m_height - a), (m_width - b))



main()
