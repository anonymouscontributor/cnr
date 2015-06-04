'''
A collection of Domain classes for the cnr problem.

@date: May 6, 2015
'''

import numpy as np
from scipy.misc import factorial
from cvxopt import solvers, matrix, spdiag, spmatrix

class Domain(object):
    """ Base class for domains """

    def iselement(self, points):
        """ Returns boolean array of the same dimension as points.
            "True" if point is contained in domain and "False" otherwise """
        raise NotImplementedError

    def compute_parameters(self):
        """ Computes diameter and volume of the domain for later use """
        raise NotImplementedError

    def bbox(self):
        """ Computes a Bounding Box (a Rectangle object) """
        raise NotImplementedError

    def union(self, domain):
        """ Returns a Domain object that is the union of the original
        object with dom (also a Domain object) """

    def isconvex(self):
        """ Returns True if the domain is convex, and False otherwise """
        return self.cvx

    def project(self, points):
        """ Projects the points in 'points' onto the domain (i.e. returns
            x* = argmin_{s \in domain} ||s-x||_2^2 """
        raise NotImplementedError


class nBox(Domain):
    """ A rectangular box in n dimensions """

    def __init__(self, bounds):
        """ Construct an nBox. bounds is a list of n tuples, where the i-th tuple
            gives the lower and upper bound of the box in dimension i """
        self.n = len(bounds)
        self.cvx = True
        self.verts = None
        self.bounds = bounds
        self.diameter, self.volume, self.v = self.compute_parameters()

    def iselement(self, points):
        """ Returns boolean array with length points.shape[0]. Element is
            "True" if point is contained in domain and "False" otherwise """
        return np.all(np.array([(points[:,i]>=self.bounds[i][0]) &
                                (points[:,i]<=self.bounds[i][1])
                                for i in range(self.n)]), axis=0)

    def compute_parameters(self):
        """ Computes diameter, volume and uniformity parameter v of the domain for later use """
        diameter = np.sqrt(np.sum([(self.bounds[i][1] - self.bounds[i][0])**2 for i in range(self.n)]))
        volume = np.array([self.bounds[i][1] - self.bounds[i][0] for i in range(self.n)]).prod()
        return diameter, volume, 1.0

    def compute_Dmu(self, mu):
        """ Computes D_mu, i.e. sup_{s in S} ||s-mu||_2^2 """
        mu = np.array(mu, ndmin=2)
        # first check that mu is contained in the domain
        if not self.iselement(mu):
            raise Exception('mu must be an element of the domain')
        return np.max([np.linalg.norm(mu - vertex) for vertex in self.vertices()])

    def bbox(self):
        """ An nBox is its own Bounding Box """
        return self

    def vertices(self):
        """ Returns the vertices of the nBox """
        if self.verts is None:
            bounds = [np.asarray(bnd) for bnd in self.bounds]
            shape = (len(arr) for arr in bounds)
            idxs = np.indices(shape, dtype=int)
            idxs = idxs.reshape(len(bounds), -1).T
            self.verts = np.array([bnd[idxs[:, n]] for n,bnd in enumerate(bounds)]).T
        return self.verts

    def grid(self, N):
        """ Returns a uniform grid with at least N gridpoints """
        Z = (np.prod([self.bounds[i][1] - self.bounds[i][0] for i in range(self.n)]))**(1/self.n)
        Ns = [np.ceil((self.bounds[i][1] - self.bounds[i][0])/Z*N**(1/self.n)) for i in range(self.n)]
        grids = [np.linspace(self.bounds[i][0], self.bounds[i][1], Ns[i]) for i in range(self.n)]
        return np.vstack(np.meshgrid(*grids)).reshape(self.n,-1).T

    def sample_uniform(self, N):
        """ Draws N samples uniformly from the nBox """
        np.random.seed()
        return np.concatenate([np.random.uniform(low=self.bounds[i][0], high=self.bounds[i][1],
                                                 size=(N,1)) for i in range(self.n)], axis=1)

    def project(self, points):
        """ Projects the points in 'points' onto the domain (i.e. returns
            x* = argmin_{s \in domain} ||s-x||_2^2. For an nBox this
            is straightforward and can be done efficiently. """
        return np.array([project_coordinate(points[:,i], bounds) for i,bounds in enumerate(self.bounds)]).T

    def gen_project(self, points, mats):
        """ Performs a generalized projection of the points in 'points' onto the
            domain, returning x* = argmin_{s \in domain} (s-x)^T*A*(s-x). """
        N = points.shape[0]
        P = spdiag([matrix(2*mat, tc='d') for mat in mats])
        q = matrix(np.einsum('ijk...,ik...->ij...', -2*mats, points).flatten(), tc='d')
        G = spmatrix([-1,1]*(self.n*N), np.arange(2*self.n*N), np.repeat(np.arange(self.n*N), 2), tc='d')
        h = matrix(np.tile(np.array([-1,1]*self.n)*np.array(self.bounds).flatten(), points.shape[0]), tc='d')
        solvers.options['show_progress'] = False
        res = solvers.qp(P, q, G, h)
        return np.array(res['x']).reshape(points.shape)


class UnionOfDisjointnBoxes(Domain):
    """ Domain consisting of a number of p/w disjoint nBoxes """

    def __init__(self, nboxes):
        """ Constructor """
        self.nboxes = nboxes
        self.cvx = False
        self.verts = None
        self.n = nboxes[0].n
        self.diameter, self.volume, self.v = self.compute_parameters()

    def iselement(self, points):
        """ Returns boolean array with length points.shape[0]. Element is
            "True" if point is contained in domain and "False" otherwise """
        return np.any([nbox.iselement(points) for nbox in self.nboxes], axis=0)

    def compute_parameters(self):
        """ Computes diameter and volume of the domain as well as a lower bound of
            the uniform fatness constant v for later use (the bound on v is tight
            if any two rectangles are a strictly positive distance apart). """
        # since rectangles are disjoint, the volume is just the sum of the
        # individual volumes and v is the minimum of the individual vs
        bbox = self.bbox()
        bbox_diameter = bbox.compute_parameters()[0]
        volumes = [nbox.volume for nbox in self.nboxes]
        volume = np.sum(volumes)
        return bbox_diameter, volume, np.min(volumes)/volume

    def bbox(self):
        """ Returns the bounding nBox """
        lower = np.array([[self.nboxes[i].bounds[j][0] for j in range(self.n)]
                          for i in range(len(self.nboxes))]).min(axis=0)
        upper = np.array([[self.nboxes[i].bounds[j][1] for j in range(self.n)]
                          for i in range(len(self.nboxes))]).max(axis=0)
        bounds = [(low, high) for low,high in zip(lower, upper)]
        return nBox(bounds)


    def compute_Dmu(self, mu):
        """ Computes D_mu, i.e. sup_{s in S} ||s-mu||_2^2 """
        mu = np.array(mu, ndmin=2)
        # first check that mu is contained in the domain
        if not self.iselement(mu):
            raise Exception('mu must be an element of the domain')
        return np.max([np.linalg.norm(mu - vertex) for vertex in self.vertices()])

    def vertices(self):
        """ Returns the vertices of the UnionOfDisjointnBoxes """
        if self.verts is None:
            self.verts = np.vstack([nbox.vertices() for nbox in self.nboxes])
        return self.verts

    def grid(self, N):
        """ Returns a uniform grid with at least N grid points """
        volumes = np.array([nbox.volume for nbox in self.nboxes])
        weights = volumes/np.sum(volumes)
        return np.vstack([nbox.grid(weight*N) for nbox,weight
                          in zip(self.nboxes, weights)])

    def sample_uniform(self, N):
        """ Draws N samples uniformly from the union of disjoint nBoxes """
        volumes = [nbox.volume for nbox in self.nboxes]
        weights = volumes/np.sum(volumes)
        np.random.seed()
        select = np.random.choice(np.arange(len(volumes)), p=weights, size=N)
        samples = np.array([nbox.sample_uniform(N) for nbox in self.nboxes])
        return samples[select, np.arange(N)]

    def project(self, points):
        """ Projects the points in 'points' onto the domain (i.e. returns
            x* = argmin_{s \in domain} ||s-x||_2^2. For a union of disjoint
            nBoxes we can just project on each nBox and take the point
            with the minimum distance. """
        proj_points = np.array([nbox.project(points) for nbox in self.nboxes])
        dists = np.array([np.linalg.norm(points - proj, axis=1) for proj in proj_points])
        return proj_points[np.argmin(dists, 0), np.arange(proj_points.shape[1])]

    def gen_project(self, points, mats):
        """ Performs a generalized projection of the points in 'points' onto the
            domain, returning x* = argmin_{s \in domain} (s-x)^T*A*(s-x). For a union
            of disjoint nBoxes we can just project on each nBox and take the point
            with the minimum distance. """
        proj_points = np.array([nbox.project(points) for nbox in self.nboxes])
        dists = np.array([np.einsum('ij,ij->i',np.einsum('ijk...,ik...->ij...', mats, proj-points), proj-points) for proj in proj_points])
        return proj_points[np.argmin(dists, 0), np.arange(proj_points.shape[1])]


class DifferenceOfnBoxes(Domain):
    """ Domain consisting of an nBox (the 'outer' box) from which
        multiple (disjoint) nBoxes are subtracted """

    def __init__(self, outer, inner):
        """ Constructor """
        violations = [((obnd[0]>ibnd[1]) | (obnd[1]<ibnd[1]))
                      for nbox in inner for obnd,ibnd in zip(outer.bounds, nbox.bounds)]
        if np.any(violations):
            raise Exception('All nBoxes in "inner" must be contained inside the outer nBox')
        self.outer = outer
        self.inner = inner
        self.cvx = False
        self.verts = None
        self.n = outer.n
        self.diameter, self.volume, self.v = self.compute_parameters()

    def iselement(self, points):
        """ Returns boolean array with length points.shape[0]. Element is
            "True" if point is contained in domain and "False" otherwise """
        in_inner = np.any([nbox.iselement(points) for nbox in self.inner], axis=0)
        return self.outer.iselement(points) & np.logical_not(in_inner)

    def compute_parameters(self):
        """ Computes diameter and volume of the domain as well as a lower bound of
            the uniform fatness constant v for later use (the bound on v is tight
            if any two rectangles are a strictly positive distance apart). """
        # since the inner rectangles are disjoint, the volume is just the volume of
        # the outer nBox minus the sum of the volumes of the inner nBoxes
        # Unfortunately determining v is more difficult ...
        diameter, volume_outer, v_outer = self.outer.compute_parameters()
        volume = volume_outer - np.sum([nbox.volume for nbox in self.inner])
        return diameter, volume, None

    def bbox(self):
        """ Returns the bounding nBox, which is just the outer nBox. """
        return self.outer

    def compute_Dmu(self, mu):
        """ Computes D_mu, i.e. sup_{s in S} ||s-mu||_2^2. Here it suffices to
            just check the vertices of the outer nBox. """
        return self.outer.compute_Dmu(mu)

    def vertices(self):
        """ Returns an array with the the vertices of the DifferenceOfnBoxes.
            This is just the union of the set of vertices from the individual nBoxes. """
        return np.vstack([self.outer.vertices(), np.vstack([nbox.vertices() for nbox in self.inner])])

    def grid(self, N):
        """ Returns a uniform grid with at least N grid points. This implementation is
            based on inflating the number of gridpoints for the grid on the outer nBox,
            and then returning only points that do not fall inside the inner nBoxes. """
        full_grid = self.outer.grid(self.outer.volume/self.volume*N)
        return full_grid[self.iselement(full_grid)]

    def sample_uniform(self, N):
        """ Draws N samples uniformly from the DifferenceOfnBoxes. This can be achieved
            by simple rejection sampling, i.e. by sampling uniformly from the outer nBox
            and discarding samples tha fall inside the inner nBoxes. This is obviously
            not very efficient and should be avoided if an additive decomposition
            of the DifferenceOfnBoxes is available. """
        samples_outer = self.outer.sample_uniform(self.outer.volume/self.volume*N)
        samples = samples_outer[self.iselement(samples_outer)]
        while len(samples) < N:
            newsamples = self.outer.sample_uniform(N/(N-len(samples))*self.outer.volume/self.volume*N)
            samples = np.vstack((samples, newsamples[self.iselement(newsamples)]))
        return samples[0:N]

    def to_UoDnB(self):
        if (self.n == 2) and (len(self.inner) == 1):
            bnds_inner, bnds_outer = self.inner[0].bounds, self.outer.bounds
            bounds = [[(bnds_outer[0][0], bnds_inner[0][0]), bnds_outer[1]],
                      [(bnds_inner[0][1], bnds_outer[0][1]), bnds_outer[1]],
                      [bnds_outer[0], (bnds_outer[1][0], bnds_inner[1][0])],
                      [bnds_outer[0], (bnds_inner[1][1], bnds_outer[1][1])]]
            return UnionOfDisjointnBoxes([nBox(bound) for bound in bounds])
        else:
            raise NotImplementedError('Sorry, to_nboxes for now only works in dimension 2 with a single inner box')

    def project(self, points):
        """ Projects the points in 'points' onto the domain (i.e. returns
            x* = argmin_{s \in domain} ||s-x||_2^2. For a difference of nBoxes
            """
        raise NotImplementedError


class UnitSimplex(Domain):
    """ The k-unit simplex (i.e. x_0,x_1,...,x_k s.t. x_i>=0 and sum(x_i)=1) """

    def __init__(self, k):
        self.k, self.n = k, k+1
        self.cvx = True
        self.diameter, self.volume, self.v = self.compute_parameters()

    def iselement(self, points):
        """ Returns boolean array with length points.shape[0]. Element is
            "True" if point is contained in domain and "False" otherwise """
        return np.all(0<=points, axis=1)*np.all(points<=1, axis=1)*(np.sum(points, axis=1) == True)

    def compute_parameters(self):
        """ Computes diameter, volume and uniformity parameter v of the domain for later use """
        return np.sqrt(self.n), 1/factorial(self.k), 1.0

    def bbox(self):
        """ The bounding box of a k-unit simplex is the (k+1) unit cube """
        return nBox([(0,1)]*self.n)

    def grid(self, N):
        """ Returns a uniform grid with at least N gridpoints """
        # this does not seem super straightforward
        return NotImplementedError

    def sample_uniform(self, N):
        """ Draws N samples uniformly from the k-unit simplex. This is just
            sampling from a Dirichlet distribution with all parameters equal to 1. """
        np.random.seed()
        return np.random.dirichlet([1]*self.k, N)


#######################################################################
# Some helper functions
#######################################################################

def project_coordinate(points, bounds):
    """ Projects a set of points in R (assumed to be 1-dim numpy array) onto a line segment
        given by bounds. Used for coordinate-wise projection in the projection onto an nBox """
    return (bounds[0]*(bounds[0] > points) + bounds[1]*(bounds[1] < points)
            + points*((bounds[0] <= points) & (bounds[1] >= points)))

# def negject_box(points, bounds):
#     """ Perform an inverse projection of a set of points to a box """
# #     contained = nBox.iselement(points)
# #     points_mod =
# #     return contained*points + np.logical_not(contained)*points_mod
#     lower, upper = np.array([bnd[0] for bnd in bounds]), np.array([bnd[1] for bnd in bounds])
#
#     dists = np.maximum(np.array([lower-points, points-upper]), 0)
#     return dists


def unitbox(n):
    """ Returns the n-unit cube [-0.5, 0.5]**n """
    return nBox([(-0.5, 0.5)]*n)

def hollowbox(n, ratio=0.5):
    """ Returns an n-cube centered at the origin with cut-out center,
        with total volume 1. Here ratio is the ratio of volume of the
        hollow cube to that of the outer cube. """
    a = (1 - ratio)**(-1/n)
    b = (ratio/(1-ratio))**(1/n)
    hbox = DifferenceOfnBoxes(nBox([(-0.5*a,0.5*a)]*n), [nBox([(-0.5*b,0.5*b)]*n)])
    hbox.v = 0.5*(a-b)*a**(n-1)
    return hbox

def vboxes(n, v):
    """ Returns a UnionOfDisjointnBoxes with specified v """
    a, b = (1-v)**(1/n), v**(1/n)
    return UnionOfDisjointnBoxes([nBox([(0,a),]*n), nBox([(-b,0),]*n)])

def vL(v, Npath=None, epsilon=0):
    """ Returns an L-shaped UnionOfDisjointnBoxes with specified v in dimension 2"""
    if v == 1:
        L = nBox([(0,1),(0,1)])
    else:
        a = (1+v, 1-v)
        b = (v, 1)
        L = UnionOfDisjointnBoxes([nBox([(0,b[0]), (0,b[1])]), nBox([(b[0], a[0]), (0,a[1])])])
        L.v = v
    if Npath is not None:
        if v == 1:
            path = (1-epsilon)*np.array([np.linspace(0.05, 0.95, Npath)]*2).T + epsilon*L.sample_uniform(Npath)
        else:
            N1 = np.floor(0.25*Npath)
            N2 = Npath - N1
            p1 = np.array([np.linspace(0.95*a[0], 0.5*b[0], N1), 0.5*a[1]*np.ones(N1)]).T
            p2 = np.array([0.5*b[0]*np.ones(N2), np.linspace(0.5*a[1], 0.95*b[1], N2)]).T
            path = np.concatenate([(1-epsilon)*p1 + epsilon*L.nboxes[1].sample_uniform(N1),
                                   (1-epsilon)*p2 + epsilon*L.nboxes[0].sample_uniform(N2)])
        return L, path
    else:
        return L
