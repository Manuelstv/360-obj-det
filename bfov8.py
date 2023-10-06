import cv2
import numpy as np
from numpy.linalg import norm
from skimage.io import imread

class Rotation:

    @staticmethod
    def Rx(alpha):
        return np.asarray([[1, 0, 0],
                           [0, np.cos(alpha), -np.sin(alpha)],
                           [0, np.sin(alpha), np.cos(alpha)]])

    @staticmethod
    def Ry(beta):
        return np.asarray([[np.cos(beta), 0, np.sin(beta)],
                           [0, 1, 0],
                           [-np.sin(beta), 0, np.cos(beta)]])

    @staticmethod
    def Rz(gamma):
        return np.asarray([[np.cos(gamma), -np.sin(gamma), 0],
                           [np.sin(gamma), np.cos(gamma), 0],
                           [0, 0, 1]])

class Plotting:

    @staticmethod
    def plotEquirectangular(image, kernel, step):
        # Resize the image to 1920x960
        resized_image = cv2.resize(image, (1920, 960))

        for point in kernel:
            cv2.circle(resized_image, (int(point[0]), int(point[1])), 5, (0, 165, 255), -1)

        cv2.imwrite(f'/home/mstveras/360-obj-det/img_2.png', resized_image)

if __name__ == "__main__":
    image = imread('/home/mstveras/360-obj-det/images/image_00307.jpg')
    h, w = image.shape[:2]
    step = 0

    #1234, 598, 0.8950266570383422, -0.38451785083000084, 20, 56,
    v00, u00 = 598,1234  # center of the kernel in equirectangular coordinates (pixel location)

    phi00 = (u00 - w / 2.) * ((2. * np.pi) / w)
    theta00 = -(v00 - h / 2.) * (np.pi / h)
    r = 9
    a = np.radians(56)
    d = r / (2 * np.tan(a / 2))

    p = []
    for i in range(-(r - 1) // 2, (r + 1) // 2):
        for j in range(-(r - 1) // 2, (r + 1) // 2):
            p += [np.asarray([i, j, d])]

    R = np.dot(Rotation.Ry(phi00), Rotation.Rx(theta00))
    p = np.asarray([np.dot(R, (p[ij] / norm(p[ij]))) for ij in range(r * r)])

    phi = np.asarray([np.arctan2(p[ij][0], p[ij][2]) for ij in range(r * r)])
    theta = np.asarray([np.arcsin(p[ij][1]) for ij in range(r * r)])

    u = (phi / (2 * np.pi) + 1. / 2.) * w
    v = h - (-theta / np.pi + 1. / 2.) * h

    Plotting.plotEquirectangular(image, np.vstack((u, v)).T, step)
