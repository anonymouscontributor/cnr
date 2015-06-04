'''
A collection of LossFunction classes

@date: May 7, 2015
'''

import numpy as np
import os, ctypes, uuid
from _ctypes import dlclose
from subprocess import call
from matplotlib import pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import axes3d
from scipy.linalg import orth, eigh
from scipy.integrate import nquad
from scipy.stats import uniform, gamma
from cvxopt import matrix, spmatrix, solvers
from .Domains import nBox, UnionOfDisjointnBoxes, DifferenceOfnBoxes


class LossFunction(object):
    """ Base class for LossFunctions """

    def val(self, points):
        """ Returns values of the loss function at the specified points """
        raise NotImplementedError

    def val_grid(self, x, y):
        """ Computes the value of the loss on rectangular grid (for now assume 2dim) """
        points = np.array([[a,b] for a in x for b in y])
        return self.val(points).reshape((x.shape[0], y.shape[0]))

    def plot(self, points):
        """ Creates a 3D plot of the density via triangulation of
            the density value at the points """
        if points.shape[1] != 2:
            raise Exception('Can only plot functions in dimension 2.')
        pltpoints = points[self.domain.iselement(points)]
        vals = self.val(pltpoints)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_trisurf(pltpoints[:,0], pltpoints[:,1], vals,
                        cmap=plt.get_cmap('jet'), linewidth=0.2)
        ax.set_xlabel('$s_1$'), ax.set_ylabel('$s_2$'), ax.set_zlabel('l$(z)$')
        ax.set_title('Loss')
        plt.show()
        return fig

    def grad(self, points):
        """ Returns gradient of the loss function at the specified points """
        raise NotImplementedError

    def Hessian(self, points):
        """ Returns Hessian of the loss function at the specified points """
        raise NotImplementedError

    def proj_gradient(self, x, step):
        """ Returns next iterate of a projected gradient step """
        grad_step = x - step*self.grad(x)
        return self.domain.project(grad_step)


class ZeroLossFunction(LossFunction):
    """ An zero loss function in n dimensions (for coding consistency) """

    def __init__(self, domain):
        """ ZeroLossFunction with l(s) = 0. """
        self.domain = domain
        self.desc = 'Zero'

    def val(self, points):
        return np.zeros(points.shape[0])

    def max(self):
        return 0

    def min(self):
        return 0

    def grad(self, points):
        return np.zeros_like(points)

    def __add__(self, lossfunc):
        """ Add a loss function object to the ZeroLossFunction """
        return lossfunc

    def norm(self, p):
        return 0

    def gen_ccode(self):
        return ['double f(int n, double args[n]){\n',
                '   double loss = 0.0;\n']



class AffineLossFunction(LossFunction):
    """ An affine loss function in n dimensions """

    def __init__(self, domain, a, b):
        """ AffineLossFunction of the form l(s) = <a,s> + b,
            where a is a vector in R^n and b is a scalar. """
        if not len(a) == domain.n:
            raise Exception('Dimension of a must be dimension of domain!')
        self.domain = domain
        self.a, self.b = np.array(a), b
        self.desc = 'Affine'

    def val(self, points):
        return np.dot(points, self.a) + self.b

    def max(self, grad=False):
        """ Compute the maximum of the loss function over the domain. """
        if isinstance(self.domain, nBox):
            M = self.b + np.sum([np.maximum(self.a*bnd[0], self.a*bnd[1]) for bnd in self.domain.bounds])
        elif isinstance(self.domain, UnionOfDisjointnBoxes):
            M = self.b + np.max([np.sum([np.maximum(self.a*bnd[0], self.a*bnd[1]) for bnd in nbox.bounds]) for nbox in self.domain.nboxes])
        elif isinstance(self.domain, DifferenceOfnBoxes):
            M = self.b + np.sum([np.maximum(self.a*bnd[0], self.a*bnd[1]) for bnd in self.domain.outer.bounds])
        else:
            raise Exception(('Sorry, for now only nBox, UnionOfDisjointnBoxes and DifferenceOfnBoxes '
                             + 'are supported for computing minimum and maximum of AffineLossFunctions'))
        if grad:
            return M, np.linalg.norm(self.a, 2)
        else:
            return M

    def min(self, argmin=False):
        """ Compute the minimum and maximum of the loss function over the domain.
            This assumes that the domain is an nBox. """
        if (isinstance(self.domain, nBox) or isinstance(self.domain, UnionOfDisjointnBoxes)
            or isinstance(self.domain, DifferenceOfnBoxes)):
            verts = self.domain.vertices()
            vals = self.val(verts)
            idx = np.argmin(vals)
            if argmin:
                return vals[idx], verts[idx]
            else:
                return vals[idx]
            return np.min(self.val(self.domain.vertices()))
        else:
            raise Exception(('Sorry, for now only nBox, UnionOfDisjointnBoxes and DifferenceOfnBoxes '
                             + 'are supported for computing minimum and maximum of AffineLossFunctions'))

    def grad(self, points):
        """ Returns the gradient of the AffineLossFunction (equal at all points) """
        return np.repeat(np.array(self.a, ndmin=2), points.shape[0], axis=0)

    def __add__(self, affine2):
        """ Add two AffineLossFunction objects (assumes that both functions
            are defined over the same domain. """
        if isinstance(affine2, AffineLossFunction):
            return AffineLossFunction(self.domain, self.a + affine2.a, self.b + affine2.b)
        else:
            raise Exception('So far can only add two affine loss functions!')

    def norm(self, p, **kwargs):
        """ Computes the p-Norm of the loss function over the domain """
        if np.isinf(p):
            return self.max()
        if isinstance(self.domain, nBox):
            nboxes = [self.domain]
        elif isinstance(self.domain, UnionOfDisjointnBoxes):
            nboxes = self.domain.nboxes
        else:
            raise Exception('Sorry, so far only nBox and UnionOfDisjointnBoxes are supported!')
        if p == 1:
            return np.sum([nbox.volume*(self.b + 0.5**np.sum([a*(bnd[1]+bnd[0]) for a,bnd in zip(self.a, nbox.bounds)]))
                           for nbox in nboxes])
        else:
            ccode = ['#include <math.h>\n\n',
                     'double a[{}] = {{{}}};\n\n'.format(self.domain.n, ','.join(str(ai) for ai in self.a)),
                     'double f(int n, double args[n]){\n',
                     '   int i;\n',
                     '   double loss = {};\n'.format(self.b),
                     '   for (i=0; i<{}; i++){{\n'.format(self.domain.n),
                     '     loss += a[i]*args[i];}\n',
                     '   return pow(fabs(loss), {});\n'.format(p),
                     '   }']
            ranges = [nbox.bounds for nbox in nboxes]
            return ctypes_integrate(ccode, ranges, **kwargs)**(1/p)

    def gen_ccode(self):
        return ['double a[{}] = {{{}}};\n'.format(self.domain.n, ','.join(str(a) for a in self.a)),
                'double f(int n, double args[n]){\n',
                '   double nu = *(args + {});\n'.format(self.domain.n),
                '   int i;\n',
                '   double loss = {};\n'.format(self.b),
                '   for (i=0; i<{}; i++){{\n'.format(self.domain.n),
                '     loss += a[i]*(*(args + i));\n',
                '     }\n']



class QuadraticLossFunction(LossFunction):
    """ Loss given by l(s) = 0.5 (s-mu)'Q(s-mu) + c, with Q>0 and c>= 0.
        This assumes that mu is inside the domain! """

    def __init__(self, domain, mu, Q, c):
        if not domain.n == len(mu):
            raise Exception('Dimension of mu must be dimension of domain!')
        if not domain.n == Q.shape[0]:
            raise Exception('Dimension of Q must be dimension of domain!')
        if not Q.shape[0] == Q.shape[1]:
            raise Exception('Matrix Q must be square!')
        self.domain, self.mu, self.Q, self.c = domain, mu, Q, c
        self.desc = 'Quadratic'
        # implement computation of Lipschitz constant. Since gradient is
        # linear, we can just look at the norm on the vertices
#         self.L = self.computeL()

    def val(self, points):
        x = points - self.mu
        return 0.5*np.sum(np.dot(x,self.Q)*x, axis=1)  + self.c

    def max(self, grad=False):
        """ Compute the maximum of the loss function over the domain. """
        if grad:
            raise NotImplementedError
        try:
            return self.bounds['max']
        except (KeyError, AttributeError) as e:
            if (isinstance(self.domain, nBox) or isinstance(self.domain, UnionOfDisjointnBoxes)
                or isinstance(self.domain, DifferenceOfnBoxes)):
                if not isPosDef(self.Q):
                    M = np.max(self.val(self.domain.grid(10000))) # this is a hack
                else:
                    M = np.max(self.val(self.domain.vertices()))
                try:
                    self.bounds['max'] = M
                except AttributeError:
                    self.bounds = {'max':M}
                return self.bounds['max']
            else:
                raise Exception(('Sorry, for now only nBox, UnionOfDisjointnBoxes and DifferenceOfnBoxes '
                                 + 'are supported for computing minimum and maximum of AffineLossFunctions'))

    def min(self, argmin=False, **kwargs):
        """ Compute the minimum of the loss function over the domain. """
        if isPosDef(self.Q):
            if self.domain.iselement(np.array(self.mu, ndmin=2)):
                minval, smin = self.c, self.mu
            elif (isinstance(self.domain, nBox) or isinstance(self.domain, UnionOfDisjointnBoxes)):
                minval, smin = self.find_boxmin(argmin=True)
            else:
                grid = self.domain.grid(50000)
                vals = self.val(grid)
                idxmin = np.argmin(vals)
                minval, smin = vals[idxmin], grid[idxmin]
        else:
            grid = self.domain.grid(50000)
            vals = self.val(grid)
            idxmin = np.argmin(vals)
            minval, smin = vals[idxmin], grid[idxmin]
        if argmin:
            return minval, smin
        else:
            return minval

    def find_boxmin(self, argmin=False):
        if isinstance(self.domain, nBox):
            bounds = [self.domain.bounds]
        elif isinstance(self.domain, UnionOfDisjointnBoxes):
            bounds = [nbox.bounds for nbox in self.domain.nboxes]
        else:
            raise Exception('Boxmin only works on nBox or UnionOfDisjointnBoxes')
        n = self.domain.n
        P = matrix(self.Q, tc='d')
        q = matrix(-np.dot(self.Q, self.mu), tc='d')
        G = spmatrix([-1,1]*n, np.arange(2*n), np.repeat(np.arange(n), 2), tc='d')
        hs = [matrix(np.array([-1,1]*n)*np.array(bound).flatten(), tc='d') for bound in bounds]
        solvers.options['show_progress'] = False
        results = [solvers.qp(P, q, G, h) for h in hs]
        smins = np.array([np.array(res['x']).flatten() for res in results])
        vals = self.val(smins)
        idxmin = np.argmin(vals)
        if argmin:
            return vals[idxmin], smins[idxmin]
        else:
            return vals[idxmin]

    def grad(self, points):
        return np.transpose(np.dot(self.Q, np.transpose(points - self.mu)))

    def Hessian(self, points):
        return np.array([self.Q,]*points.shape[0])

    def __add__(self, quadloss):
        Qtilde = self.Q + quadloss.Q
        btilde = - np.dot(self.mu, self.Q) - np.dot(quadloss.mu, quadloss.Q)
        mutilde = -np.linalg.solve(Qtilde, btilde)
#         print('min eval of Q: {}, ||mutilde||: {}'.format(np.min(np.abs(np.linalg.eigvalsh(Qtilde))), np.linalg.norm(mutilde,2)))
        ctilde = 0.5*(2*self.c + np.dot(self.mu, np.dot(self.Q, self.mu))
                      + 2*quadloss.c + np.dot(quadloss.mu, np.dot(quadloss.Q, quadloss.mu))
                      + np.dot(btilde, mutilde))
        return QuadraticLossFunction(self.domain, mutilde, Qtilde, ctilde)

    def norm(self, p, **kwargs):
        """ Computes the p-Norm of the loss function over the domain """
        nboxes = None
        if isinstance(self.domain, nBox):
            nboxes = [self.domain]
        elif isinstance(self.domain, UnionOfDisjointnBoxes):
            nboxes = self.domain.nboxes
        elif isinstance(self.domain, DifferenceOfnBoxes):
            if (self.domain.inner) == 1:
                nboxes = self.domain.to_nboxes()
        if nboxes is None:
            raise Exception('Sorry, so far only nBox, UnionOfDisjointnBoxes and hollowboxes are supported!')
        if np.isinf(p):
            return self.max()
        else:
            ccode = ['#include <math.h>\n\n',
                     'double Q[{}][{}] = {{{}}};\n'.format(self.domain.n, self.domain.n, ','.join(str(q) for row in self.Q for q in row)),
                     'double mu[{}] = {{{}}};\n'.format(self.domain.n, ','.join(str(m) for m in self.mu)),
                     'double c = {};\n\n'.format(self.c),
                     'double f(int n, double args[n]){\n',
                     '   double nu = *(args + {});\n'.format(self.domain.n),
                     '   int i,j;\n',
                     '   double loss = c;\n',
                     '   for (i=0; i<{}; i++){{\n'.format(self.domain.n),
                     '     for (j=0; j<{}; j++){{\n'.format(self.domain.n),
                     '       loss += 0.5*Q[i][j]*(args[i]-mu[i])*(args[j]-mu[j]);\n',
                     '       }\n',
                     '     }\n',
                     '   return pow(fabs(loss), {});\n'.format(p),
                     '   }']
            ranges = [nbox.bounds for nbox in nboxes]
            return ctypes_integrate(ccode, ranges, **kwargs)**(1/p)

    def gen_ccode(self):
        return ['double Q[{}][{}] = {{{}}};\n'.format(self.domain.n, self.domain.n, ','.join(str(q) for row in self.Q for q in row)),
                'double mu[{}] = {{{}}};\n'.format(self.domain.n, ','.join(str(m) for m in self.mu)),
                'double c = {};\n\n'.format(self.c),
                'double f(int n, double args[n]){\n',
                '   double nu = *(args + {});\n'.format(self.domain.n),
                '   int i,j;\n',
                '   double loss = c;\n',
                '   for (i=0; i<{}; i++){{\n'.format(self.domain.n),
                '     for (j=0; j<{}; j++){{\n'.format(self.domain.n),
                '       loss += 0.5*Q[i][j]*(args[i]-mu[i])*(args[j]-mu[j]);\n',
                '       }\n',
                '     }\n']



class PolynomialLossFunction(LossFunction):
    """ A polynomial loss function in n dimensions of arbitrary order,
        represented in the basis of monomials """

    def __init__(self, domain, coeffs, exponents, partials=False):
        """ Construct a PolynomialLossFunction that is the sum of M monomials.
            coeffs is an M-dimensional array containing the coefficients of the
            monomials, and exponents is a list of n-tuples of length M, with
            the i-th tuple containing the exponents of the n variables in the monomial.

            For example, the polynomial l(x) = 3*x_1^3 + 2*x_1*x_3 + x2^2 + 2.5*x_2*x_3 + x_3^3
            in dimension n=3 is constructed using
                coeffs = [3, 2, 1, 2.5, 1] and
                exponents = [(3,0,0), (1,0,1), (0,2,0), (0,1,1), (0,0,3)]
        """
        if not len(coeffs) == len(exponents):
            raise Exception('Size of coeffs must be size of exponents along first axis!')
        if not len(exponents[0]) == domain.n:
            raise Exception('Dimension of elements of coeffs must be dimension of domain!')
        self.domain = domain
        # remove zero coefficients (can be generated by differentiation)
        self.coeffs, self.exponents = coeffs, exponents
        self.m = len(self.coeffs)
        self.polydict = {exps:coeff for coeff,exps in zip(self.coeffs,self.exponents)}
        self.desc = 'Polynomial'
        if partials:
            self.partials = self.compute_partials()

    def val(self, points):
        monoms = np.array([points**exps for exps in self.polydict.keys()]).prod(2)
        return 0 + np.sum([monom*coeff for monom,coeff in zip(monoms, self.polydict.values())], axis=0)

    def max(self, grad=False):
        """ Compute the maximum of the loss function over the domain. If grad=True
            also compute the maximum 2-norm of the gradient over the domain.
            For now resort to stupid gridding. """
        grid = self.domain.grid(10000)
        M = np.max(self.val(grid)) # this is a hack
        if grad:
            return M, np.max(np.array([np.linalg.norm(self.grad(grid), 2, axis=1)])) # this is a hack
        else:
            return M

    def min(self, argmin=False):
        """ Compute the maximum of the loss function over the domain.
            For now resort to stupid gridding. """
        grid = self.domain.grid(50000)
        vals = self.val(grid)
        idxmin = np.argmin(vals)
        minval, smin = vals[idxmin], grid[idxmin]
        if argmin:
            return minval, smin
        else:
            return minval

    def grad(self, points):
        """ Computes the gradient of the PolynomialLossFunction at the specified points """
        try:
            partials = self.partials
        except AttributeError:
            self.partials = self.compute_partials()
            partials = self.partials
        return np.array([partials[i].val(points) for i in range(self.domain.n)]).T

    def compute_partials(self):
        partials = []
        for i in range(self.domain.n):
            coeffs = np.array([coeff*expon[i] for coeff,expon in zip(self.coeffs, self.exponents) if expon[i]!=0])
            exponents = [expon[0:i]+(expon[i]-1,)+expon[i+1:] for expon in self.exponents if expon[i]!=0]
            if len(coeffs) == 0:
                partials.append(ZeroLossFunction(self.domain))
            else:
                partials.append(PolynomialLossFunction(self.domain, coeffs, exponents))
        return partials

    def __add__(self, poly2):
        """ Add two PolynomialLossFunction objects (assumes that both polynomials
            are defined over the same domain. """
        newdict = self.polydict.copy()
        for exps, coeff in poly2.polydict.items():
            try:
                newdict[exps] = newdict[exps] + coeff
            except KeyError:
                newdict[exps] = coeff
        return PolynomialLossFunction(self.domain, list(newdict.values()), list(newdict.keys()))

    def __mul__(self, scalar):
        """ Multiply a PolynomialLossFunction object with a scalar """
        return PolynomialLossFunction(self.domain, scalar*np.array(self.coeffs), self.exponents)

    def __rmul__(self, scalar):
        """ Multiply a PolynomialLossFunction object with a scalar """
        return self.__mul__(scalar)

    def norm(self, p, **kwargs):
        """ Computes the p-Norm of the loss function over the domain """
        if isinstance(self.domain, nBox):
            nboxes = [self.domain]
        elif isinstance(self.domain, UnionOfDisjointnBoxes):
            nboxes = self.domain.nboxes
        else:
            raise Exception('Sorry, so far only nBox and UnionOfDisjointnBoxes are supported!')
        if np.isinf(p):
            return self.max()
        else:
            ccode = ['#include <math.h>\n\n',
                     'double c[{}] = {{{}}};\n'.format(self.m, ','.join(str(coeff) for coeff in self.coeffs)),
                     'double e[{}] = {{{}}};\n\n'.format(self.m*self.domain.n, ','.join(str(xpnt) for xpntgrp in self.exponents for xpnt in xpntgrp)),
                     'double f(int n, double args[n]){\n',
                     '   double nu = *(args + {});\n'.format(self.domain.n),
                     '   int i,j;\n',
                     '   double mon;\n',
                     '   double loss = 0.0;\n',
                     '   for (i=0; i<{}; i++){{\n'.format(self.m),
                     '     mon = 1.0;\n',
                     '     for (j=0; j<{}; j++){{\n'.format(self.domain.n),
                     '       mon = mon*pow(args[j], e[i*{}+j]);\n'.format(self.domain.n),
                     '       }\n',
                     '     loss += c[i]*mon;}\n',
                     '   return pow(fabs(loss), {});\n'.format(p),
                     '   }']
            ranges = [nbox.bounds for nbox in nboxes]
            return ctypes_integrate(ccode, ranges, **kwargs)**(1/p)

    def gen_ccode(self):
        return ['double c[{}] = {{{}}};\n'.format(self.m, ','.join(str(coeff) for coeff in self.coeffs)),
                'double e[{}] = {{{}}};\n\n'.format(self.m*self.domain.n, ','.join(str(xpnt) for xpntgrp in self.exponents for xpnt in xpntgrp)),
                'double f(int n, double args[n]){\n',
                '   double nu = *(args + {});\n'.format(self.domain.n),
                '   int i,j;\n',
                '   double mon;\n',
                '   double loss = 0.0;\n',
                '   for (i=0; i<{}; i++){{\n'.format(self.m),
                '     mon = 1.0;\n',
                '     for (j=0; j<{}; j++){{\n'.format(self.domain.n),
                '       mon = mon*pow(args[j], e[i*{}+j]);\n'.format(self.domain.n),
                '       }\n',
                '     loss += c[i]*mon;\n',
                '     }\n']


#######################################################################
# Some helper functions
#######################################################################


def ctypes_integrate(ccode, ranges, tmpfolder='libs/', **kwargs):
    tmpfile = '{}{}'.format(tmpfolder, str(uuid.uuid4()))
    with open(tmpfile+'.c', 'w') as file:
        file.writelines(ccode)
    call(['gcc', '-shared', '-o', tmpfile+'.dylib', '-fPIC', tmpfile+'.c'])
    lib = ctypes.CDLL(tmpfile+'.dylib')
    lib.f.restype = ctypes.c_double
    lib.f.argtypes = (ctypes.c_int, ctypes.c_double)
    result = np.sum([nquad(lib.f, rng)[0] for rng in ranges])
    dlclose(lib._handle) # this is to release the lib, so we can import the new version
    try:
        os.remove(tmpfile+'.c') # clean up
        os.remove(tmpfile+'.dylib') # clean up
    except FileNotFoundError: pass
    return result


def create_random_gammas(covs, Lbnd, dist=uniform()):
    """ Creates a random scaling factor gamma for each of the covariance matrices
        in the array like 'covs', based on the Lipschitz bound L. Here dist is a 'frozen'
        scipy.stats probability distribution supported on [0,1] """
    # compute upper bound for scaling
    gammas = np.zeros(covs.shape[0])
    for i,cov in enumerate(covs):
        lambdamin = eigh(cov, eigvals=(0,0), eigvals_only=True)
        gammamax = np.sqrt(lambdamin*np.e)*Lbnd
        gammas[i] = gammamax*dist.rvs(1)
    return gammas


def create_random_Sigmas(n, N, L, M, dist=gamma(2, scale=2)):
    """ Creates N random nxn covariance matrices s.t. the Lipschitz
        constants are uniformly bounded by Lbnd. Here dist is a 'frozen'
        scipy.stats probability distribution supported on R+"""
    # compute lower bound for eigenvalues
    lambdamin = ((2*np.pi)**n*np.e*L**2/M**2)**(-1.0/(n+1))
    Sigmas = []
    for i in range(N):
        # create random orthonormal matrix
        V = orth(np.random.uniform(size=(n,n)))
        # create n random eigenvalues from the distribution and shift them by lambdamin
        lambdas = lambdamin + dist.rvs(n)
        Sigma = np.zeros((n,n))
        for lbda, v in zip(lambdas, V):
            Sigma = Sigma + lbda*np.outer(v,v)
        Sigmas.append(Sigma)
    return np.array(Sigmas)


def create_random_Q(domain, mu, L, M, pd=True, H=0, dist=uniform()):
    """ Creates random symmetric nxn matrix s.t. the Lipschitz constant of the resulting
        quadratic function is bounded by L. Here M is the uniform bound on the maximal loss.
        If pd is True, then the matrix is pos. definite and H is a lower bound on the
        eigenvalues of Q.  Finally, dist is a 'frozen' scipy.stats probability
        distribution supported on [0,1].  """
    n = domain.n
    # compute upper bound for eigenvalues
    Dmu = domain.compute_Dmu(mu)
    lambdamax = np.min((L/Dmu, 2*M/Dmu**2))
    # create random orthonormal matrix
    V = orth(np.random.uniform(size=(n,n)))
    # create n random eigenvalues from the distribution dist, scale them by lambdamax
    if pd:
        lambdas = H + (lambdamax-H)*dist.rvs(n)
    else:
        # randomly assign pos. and negative values to the evals
        lambdas = np.random.choice((-1,1), size=n)*lambdamax*dist.rvs(n)
    return np.dot(V, np.dot(np.diag(lambdas), V.T)), lambdamax


def create_random_Cs(covs, dist=uniform()):
    """ Creates a random offset C for each of the covariance matrices in the
        array-like 'covs'. Here dist is a 'frozen' scipy.stats probability
        distribution supported on [0,1] """
    C = np.zeros(covs.shape[0])
    for i,cov in enumerate(covs):
        pmax = ((2*np.pi)**covs.shape[1]*np.linalg.det(cov))**(-0.5)
        C[i] = np.random.uniform(low=pmax, high=1+pmax)
    return C


def random_AffineLosses(dom, L, T, d=2):
    """ Creates T random L-Lipschitz AffineLossFunction over domain dom,
        and returns uniform bound M. For now sample the a-vector uniformly
        from the n-ball. Uses random samples of Beta-like distributions as
        described in the funciton sample_Bnrd. """
    lossfuncs, Ms = [], []
    asamples = sample_Bnrd(dom.n, L, d, T)
    for a in asamples:
        lossfunc = AffineLossFunction(dom, a, 0)
        lossmin, lossmax = lossfunc.min(), lossfunc.max()
        lossfunc.b = - lossmin
        lossfuncs.append(lossfunc)
        Ms.append(lossmax - lossmin)
    return lossfuncs, np.max(Ms)

def sample_Bnrd(n, r, d, N):
    """ Draw N independent samples from the B_n(r,d) distribution
        discussed in:
        'R. Harman and V. Lacko. On decompositional algorithms for uniform
        sampling from n-spheres and n-balls. Journal of Multivariate
        Analysis, 101(10):2297 – 2304, 2010.'
    """
    Bsqrt = np.sqrt(np.random.beta(n/2, d/2, size=N))
    X = np.random.randn(N, n)
    normX = np.linalg.norm(X, 2, axis=1)
    S = X/normX[:, np.newaxis]
    return r*Bsqrt[:, np.newaxis]*S


def random_QuadraticLosses(dom, mus, L, M, pd=True, H=0, dist=uniform()):
    """ Creates T random L-Lipschitz QuadraticLossFunctions over the domain dom.
        Here mus is an iterable of means (of length T), L and M are uniform bounds
        on Lipschitzc onstant and dual norm of the losses. H is the strong convexity
        parameter (here this is a lower bound on the evals of Q). dist is a 'frozen'
        scipy distribution object from which to draw the values of the evals.
    """
    lossfuncs, Ms, lambdamaxs = [], [], []
    for mu in mus:
        Q, lambdamax = create_random_Q(dom, mu, L, M, pd, H, dist)
        lossfunc = QuadraticLossFunction(dom, mu, Q, 0)
        c = -lossfunc.min()
        lossfuncs.append(QuadraticLossFunction(dom, mu, Q, c))
        Ms.append(lossfuncs[-1].max())
        lambdamaxs.append(lambdamax)
    return lossfuncs, np.max(Ms), np.max(lambdamaxs)

def isPosDef(Q):
    """ Checks whether the numpy array Q is positive definite """
    try:
        np.linalg.cholesky(Q)
        return True
    except np.linalg.LinAlgError:
        return False

def random_PolynomialLosses(dom, T, M, L, m_max, exponents, dist=uniform(), high_ratio=False):
    """ Creates T random L-Lipschitz PolynomialLossFunctions uniformly bounded
        (in dual norm) by M, with Lipschitz constant uniformly bounded by L.
        Here exponents is a (finite) set of possible exponents.
        This is a brute force implementation and horribly inefficient.
    """
    lossfuncs = []
    while len(lossfuncs) < T:
        if high_ratio:
            weights = np.ones(len(exponents))
        else:
            weights = np.linspace(1, 10, len(exponents))
        expon = [tuple(np.random.choice(exponents, size=dom.n, p=weights/np.sum(weights)))
                 for i in range(np.random.choice(np.arange(2,m_max)))]
        if high_ratio:
            coeffs = np.array([uniform(scale=np.max(expo)).rvs(1) for expo in expon]).flatten()
        else:
            coeffs = dist.rvs(len(expon))
        lossfunc = PolynomialLossFunction(dom, coeffs, expon)
        Ml, Ll = lossfunc.max(grad=True)
        ml = lossfunc.min()
        if (Ml-ml)>0:
            scaling = dist.rvs()*np.minimum(M/(Ml-ml), L/Ll)
        else:
            scaling = 1
        lossfuncs.append(scaling*(lossfunc + PolynomialLossFunction(dom, [-ml], [(0,)*dom.n])))
    return lossfuncs
