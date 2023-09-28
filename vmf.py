from numpy import sqrt, array, log, matrix, dot, squeeze, asarray, mean, degrees, arccos
from numpy.linalg import norm, svd
from numpy.random import randn
from scipy.stats import uniform, beta 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def __rW(m, kappa):
    dim = m-1
    b = dim / (sqrt(4*kappa*kappa + dim*dim) + 2*kappa)
    x = (1-b) / (1+b)
    c = kappa*x + dim*log(1-x*x)

    y = []
    done = False
    while not done:
        z = beta.rvs(dim/2,dim/2)
        w = (1 - (1+b)*z) / (1 - (1-b)*z)
        u = uniform.rvs()
        if kappa*w + dim*log(1-x*w) - c >= log(u):
            done = True
    y.append(w)
    return array(y)

def __sampleTangentUnit(mu):
    mat = matrix(mu)
    if mat.shape[1]>mat.shape[0]:
        mat = mat.T

    U,_,_ = svd(mat)
    nu = matrix(randn(mat.shape[0])).T
    x = dot(U[:,1:],nu[1:,:])
    return (x/norm(x)).T

def vMFV0(mu, kappa):
    dim = len(mu)
    w = __rW(dim, kappa)
    v = __sampleTangentUnit(mu)
    r = (sqrt(1-w**2)*v + w*mu)
    r = r / norm(r)
    return squeeze(asarray(r))

def plot_vMF(mu, kappa, num_samples=1000):
    samples = [vMFV0(mu, kappa) for _ in range(num_samples)]
    samples = np.array(samples)
    x, y, z = samples[:, 0], samples[:, 1], samples[:, 2]

    # 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(121, projection='3d')

    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x_sphere = np.cos(u)*np.sin(v)
    y_sphere = np.sin(u)*np.sin(v)
    z_sphere = np.cos(v)
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color="grey", alpha=0.2)

    ax.scatter(x, y, z, c='b', marker='o')
    ax.scatter([mu[0]], [mu[1]], [mu[2]], color="r", s=100)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Equirectangular Projection
    ax = fig.add_subplot(122)

    lon = np.arctan2(y, x)
    lat = np.arccos(z)

    # Convert to degrees
    lon = np.degrees(lon)
    lat = np.degrees(lat)
    
    ax.scatter(lon, lat, c='b', marker='o')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    
    plt.title('vMF Distribution on Sphere and Equirectangular Projection')
    plt.show()

if __name__ == "__main__":
    mu = np.array([0, 0, 1])
    kappa = 10
    plot_vMF(mu, kappa)

