from PIL import Image
import json
import numpy as np
import matplotlib.cm as cm
from skimage.io import *
from skimage.transform import *
from numpy import *
from numpy.linalg import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Rotation:
	@staticmethod
	def Rx(alpha):
		return asarray([[1, 0, 0], [0, cos(alpha), -sin(alpha)], [0, sin(alpha), cos(alpha)]])
	@staticmethod
	def Ry(beta):
		return asarray([[cos(beta), 0, sin(beta)], [0, 1, 0], [-sin(beta), 0, cos(beta)]])
	@staticmethod
	def Rz(gamma):
		return asarray([[cos(gamma), -sin(gamma), 0], [sin(gamma), cos(gamma), 0], [0, 0, 1]])

class Plotting:
	@staticmethod
	def plotEquirectangular(image, kernel, step):
		fig = plt.figure()
		fig.set_size_inches(6,3)
		ax = plt.Axes(fig, [0., 0., 1., 1.])
		ax.set_axis_off()
		ax.set_xlim([0,image.shape[1]-1])
		fig.add_axes(ax)
		plt.imshow(image)
		plt.scatter(kernel[:,0], kernel[:,1], c='orange', s=5)
		plt.clf()
		plt.cla()
		plt.close(fig)

'''
with open('/home/mstveras/360-obj-det/annotations/7lCpD.json', 'r') as f:
    data = json.load(f)

unique_classes = set(data['class'])
num_classes = len(unique_classes)
colors = cm.rainbow(np.linspace(0, 1, num_classes))
class_to_color = {cls: colors[i] for i, cls in enumerate(unique_classes)}
'''


if __name__ == "__main__":
    image = imread('/home/mstveras/360-obj-det/images/7fB0x.jpg')
    h, w = image.shape[:2]
    step = 0
    center_x, center_y, phi, theta, h, w = 1714, 494, 2.4658229838332386, -0.044178646691106396, 80, 80
    phi00 = (center_x - w/2.) * ((2. * pi) / w)
    theta00 = -(center_y - h / 2.) * (pi / h)
	
    print(phi00,theta00)
	
    r = 3
    a = radians(30)
    d = r / (2 * tan(a / 2))
    p = []
    for i in range(-(r - 1) // 2, (r + 1) // 2):
        for j in range(-(r - 1) // 2, (r + 1) // 2):
            p += [asarray([i, j, d])]
    R = dot(Rotation.Ry(phi00), Rotation.Rx(theta00))
    p = asarray([dot(R, (p[ij] / norm(p[ij]))) for ij in range(r * r)])
    phi = asarray([arctan2(p[ij][0], p[ij][2]) for ij in range(r * r)])
    theta = asarray([arcsin(p[ij][1]) for ij in range(r * r)])
    print(phi,theta)
    u = (phi / (2 * pi) + 1. / 2.) * w
    v = h - (-theta / pi + 1. / 2.) * h
    Plotting.plotEquirectangular(image, vstack((u, v)).T, step)
