# Author: Thiago L T da Silveira
# Code based on the paper arXiv:1903.08094v2

#import cv2
import sys
from skimage.io import *
from skimage.transform import *
from numpy import *
from numpy.linalg import *
from pdb import set_trace as pause
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Rotation:

	@staticmethod
	def Rx(alpha):
		return asarray([[1, 0, 0],
						[0, cos(alpha), -sin(alpha)],
						[0, sin(alpha), cos(alpha)]])

	@staticmethod
	def Ry(beta):
		return asarray([[cos(beta), 0, sin(beta)],
						[0, 1, 0],
						[-sin(beta), 0, cos(beta)]])

	@staticmethod
	def Rz(gamma):
		return asarray([[cos(gamma), -sin(gamma), 0],
						[sin(gamma), cos(gamma), 0],
						[0, 0, 1]])
"""
class Plotting:

	@staticmethod
	def plotSpherePoints(points):
		pass

	@staticmethod
	def plotEquirectangular(image, kernel, step):
		fig = plt.figure()
		fig.set_size_inches(6, 3)
		ax = plt.Axes(fig, [0., 0., 1., 1.])
		ax.set_axis_off()
		ax.set_xlim([0, image.shape[1] - 1])
		fig.add_axes(ax)
		plt.imshow(image)
		plt.scatter(kernel[:, 0], kernel[:, 1], c='orange', s=5)
		fig.savefig('/home/mstveras/img1.png')
		plt.clf()
		plt.cla()
		plt.close(fig)

	@staticmethod
	def plot3DOnTheSphere(x):
		x = matrix(x)
		fig = plt.figure(figsize=(6, 6))
		ax = fig.add_subplot(111, projection='3d')
		u = linspace(0, pi, 30, endpoint=False)
		v = linspace(0, 2 * pi, 30, endpoint=True)
		u, v = meshgrid(u, v)
		ax.plot_wireframe(cos(u) * sin(v), sin(u) * sin(v), cos(v), color=.25 * ones(3), alpha=.0675)
		ax.scatter(squeeze(asarray(x[:, 0])), squeeze(asarray(x[:, 1])), squeeze(asarray(x[:, 2])), color='red', s=5, alpha=.5)
		ax.grid('off')
		ax.set_axis_off()
		ax.set_xlim([-1, 1])
		ax.set_ylim([-1, 1])
		ax.set_zlim([-1, 1])
		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')
		plt.show()
		plt.close()
"""

class Plotting:

    @staticmethod
    def plotSpherePoints(points):
        pass

    @staticmethod
    def plotEquirectangular(image, kernel, step):
        fig = plt.figure()
        
        # Calculate the figure size to achieve 1920x960 resolution
        dpi = 100  # You can adjust this value
        fig_width = 1920 / dpi
        fig_height = 960 / dpi
        fig.set_size_inches(fig_width, fig_height)
        
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        ax.set_xlim([0, image.shape[1] - 1])
        fig.add_axes(ax)
        plt.imshow(image)
        plt.scatter(kernel[:, 0], kernel[:, 1], c='orange', s=5)
        
        # Save the figure with the specified dpi to achieve 1920x960 resolution
        fig.savefig('/home/mstveras/360-obj-det/img1.png', dpi=dpi)
        
        plt.clf()
        plt.cla()
        plt.close(fig)

    @staticmethod
    def plot3DOnTheSphere(x):
        x = matrix(x)
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')
        u = linspace(0, pi, 30, endpoint=False)
        v = linspace(0, 2 * pi, 30, endpoint=True)
        u, v = meshgrid(u, v)
        ax.plot_wireframe(cos(u) * sin(v), sin(u) * sin(v), cos(v), color=.25 * ones(3), alpha=.0675)
        ax.scatter(squeeze(asarray(x[:, 0])), squeeze(asarray(x[:, 1])), squeeze(asarray(x[:, 2])), color='red', s=5, alpha=.5)
        ax.grid('off')
        ax.set_axis_off()
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()
        plt.close()

if __name__ == "__main__":
	image = imread('/home/mstveras/360-obj-det/images/image_00307.jpg')
	h, w = image.shape[:2]
	step = 0
	#627, 551, -1.0913761978877041, -0.23071071049800057, 4, 4
	#1210, 628, 0.8164868406985976, -0.4826926212546819, 24, 20
	v00, u00 = 1210, 628  # center of the kernel in equirectangular coordinates (pixel location)

	phi00 = (u00 - w / 2.) * ((2. * pi) / w)
	theta00 = -(v00 - h / 2.) * (pi / h)
	r = 10
	a = radians(24)
	d = r / (2 * tan(a / 2))

	p = []
	for i in range(-(r - 1) // 2, (r + 1) // 2):
		for j in range(-(r - 1) // 2, (r + 1) // 2):
			p += [asarray([i, j, d])]

	R = dot(Rotation.Ry(phi00), Rotation.Rx(theta00))
	p = asarray([dot(R, (p[ij] / norm(p[ij]))) for ij in range(r * r)])

	phi = asarray([arctan2(p[ij][0], p[ij][2]) for ij in range(r * r)])
	theta = asarray([arcsin(p[ij][1]) for ij in range(r * r)])

	u = (phi / (2 * pi) + 1. / 2.) * w
	v = h - (-theta / pi + 1. / 2.) * h

	Plotting.plotEquirectangular(image, vstack((u, v)).T, step)
	step += 1
