'''
A collection of omega-potentials for Dual Averaging

@date: May 27, 2015
'''

import numpy as np

class OmegaPotential(object):
    """ Base class for omega potentials """

    def phi(self, u):
        """ Returns phi(u), the value of the zero-potential at the points u"""
        raise NotImplementedError

    def phi_prime(self, u):
        """ Returns phi'(u), the first derivative of the zero-potential at the points u """
        raise NotImplementedError

    def phi_double_prime(self, u):
        """ Returns phi''(u), the second derivative of the zero-potential at the points u """
        raise NotImplementedError

    def phi_inv(self, u):
        """ Returns phi^{-1}(u), the inverse function of the zero-potential at the points u """
        raise NotImplementedError

    def alpha_opt(self, n):
        """ Returns the optimal decay parameter alpha in the learning rate
            eta_t = theta*t**(-alpha). Here n is the dimension of the space. """
        return 1/(2 + n*self.bounds_asymp()[1])

    def theta_opt(self, domain, M):
        """ Computes the optimal constant theta for the learning rate eta_t = theta*t**(-alpha). """
        C, epsilon = self.bounds_asymp()
        lpsi, pdual = self.l_psi()
        n, v = domain.n, domain.v
        return np.sqrt(C*lpsi*(1+n*epsilon)/M**2/v**epsilon/(2+n*epsilon))

    def phi_inv_prime(self, u):
        """ Returns phi^{-1}'(u), the first derivative of the inverse
            function of the zero-potential at the points u """
        raise NotImplementedError


class ExponentialPotential(OmegaPotential):
    """ The exponential potential, which results in Entropy Dual Averaging (HEDGE) """

    def __init__(self, desc='ExpPot'):
        """ Constructor """
        self.c_omega, self.d_omega = 1, 0
        self.desc = desc

    def phi(self, u):
        """ Returns phi(u), the value of the zero-potential at the points u"""
        return np.exp(u-1)

    def phi_prime(self, u):
        """ Returns phi'(u), the first derivative of the zero-potential at the points u """
        return self.phi(u)

    def phi_double_prime(self, u):
        """ Returns phi''(u), the second derivative of the zero-potential at the points u """
        return self.phi(u)

    def phi_inv(self, u):
        """ Returns phi^{-1}(u), the inverse function of the zero-potential at the points u """
        return 1 + np.log(u)

    def phi_inv_prime(self, u):
        """ Returns phi^{-1}'(u), the first derivative of the inverse
            function of the zero-potential at the points u """
        return 1/u

    def isconvex(self):
        """ Returns True if phitilde(u) = max(phi(u), 0) is a convex function. """
        return True

    def gen_ccode(self):
        """ Generates a c-code snippet used for fast numerical integration """
        return ['   return exp(-eta*(loss + nu));}']

    def gen_KLccode(self):
        """ Generates a c-code snippet used for fast numerical integration for KL divergence"""
        return ['   return -eta*(loss + nu)*exp(-eta*(loss + nu));}']


class IdentityPotential(OmegaPotential):
    """ The identity potential Phi(x) = x, which results in the Euclidean Projection  """

    def __init__(self, desc='IdPot', **kwargs):
        """ Constructor """
        if kwargs.get('M') is not None:
            self.M = kwargs.get('M')
        self.desc = desc

    def phi(self, u):
        """ Returns phi(u), the value of the zero-potential at the points u"""
        return (u>=0)*u

    def phi_prime(self, u):
        """ Returns phi'(u), the first derivative of the zero-potential at the points u """
        return (u>=0)*np.ones_like(u)

    def phi_double_prime(self, u):
        """ Returns phi''(u), the second derivative of the zero-potential at the points u """
        return np.zeros_like(u)

    def phi_inv(self, u):
        """ Returns phi^{-1}(u), the inverse function of the zero-potential at the points u """
        return u

    def phi_inv_prime(self, u):
        """ Returns phi^{-1}'(u), the first derivative of the inverse
            function of the zero-potential at the points u """
        return np.ones_like(u)

    def isconvex(self):
        """ Returns True if phitilde(u) = max(phi(u), 0) is a convex function. """
        return True

    def l_psi(self):
        """ Returns the strong convexity constant l_psi and the corresponding norm
            of the Csizar divergence associated with the IdentityPotential. """
        return 1, 2

    def bounds_asymp(self):
        """ Returns constants C and epsilon s.t. f_phi(x) <= C x**(1+epsilon)
            for all x >= 1. See Theorem 2 in paper for details. """
        return 0.5, 1

    def gen_ccode(self):
        """ Generates a c-code snippet used for fast numerical integration """
        return ['   double z = -eta*(loss + nu);\n',
                '   return fmax(z, 0.0);',
                '   }']

    def gen_KLccode(self):
        """ Generates a c-code snippet used for fast numerical integration for KL divergence"""
        return ['   double z = -eta*(loss + nu);\n',
                '   return fmax(z, 0.0)*log(fmax(z, 0.0));',
                '   }']



class pNormPotential(OmegaPotential):
    """ The potential phi(u) = sgn(u)*|u|**(1/(p-1)) """

    def __init__(self, p, desc='pNormPot', **kwargs):
        """ Constructor """
        if (p<=1) or (p>2):
            raise Exception('Need 1 < p <=2 !')
        self.p = p
        self.desc = desc + ', ' + r'$p={{{}}}$'.format(p)
        if kwargs.get('M') is not None:
            self.M = kwargs.get('M')

    def phi(self, u):
        """ Returns phi(u), the value of the pNorm-potential at the points u"""
        return (u>=0)*np.abs(u)**(1/(self.p - 1))

    def phi_prime(self, u):
        """ Returns phi'(u), the first derivative of the pNorm-potential at the points u """
        return (u>=0)*np.abs(u)**((2 - self.p)/(self.p - 1))/(self.p - 1)

    def phi_double_prime(self, u):
        """ Returns phi''(u), the second derivative of the pNorm-potential at the points u """
        return (u>=0)*(2 - self.p)/((self.p - 1)**2)*np.abs(u)**((3 - 2*self.p)/(self.p - 2))

    def phi_inv(self, u):
        """ Returns phi^{-1}(u), the inverse function of the pNorm-potential at the points u """
        return u**(self.p - 1)

    def phi_inv_prime(self, u):
        """ Returns phi^{-1}'(u), the first derivative of the inverse
            function of the pNorm-potential at the points u """
        return (self.p - 1)*u**(self.p - 2)

    def isconvex(self):
        """ Returns True if phitilde(u) = max(phi(u), 0) is a convex function. """
        return True

    def bounds_asymp(self):
        """ Returns constants C and epsilon s.t. f_phi(x) <= C x**(1+epsilon)
            for all x >= 1. See Theorem 2 in paper for details. """
        return 1/self.p, self.p - 1

    def l_psi(self):
        """ Returns the strong convexity constant l_psi and the corresponding norm
            of the Csizar divergence associated with the pNorm Potential. """
        return self.p-1, 2/(3-self.p)

    def gen_ccode(self):
        """ Generates a c-code snippet used for fast numerical integration """
        return ['   double z = -eta*(loss + nu);\n',
                '   if(z>0){\n',
                '     return pow(z, {});}}\n'.format(1/(self.p - 1)),
                '   else{\n',
                '     return 0.0;}\n',
                '   }']

    def gen_KLccode(self):
        """ Generates a c-code snippet used for fast numerical integration for KL divergence"""
        return ['   double z = -eta*(loss + nu);\n',
                '   if(z>0){\n',
                '     return pow(z, {0})*log(pow(z, {0}));}}\n'.format(1/(self.p - 1)),
                '   else{\n',
                '     return 0.0;}\n',
                '   }']


class FractionalLinearPotential(OmegaPotential):
    """ A fractional-linear potential formed by stitching together a
        fractional and a linear function """

    def __init__(self, gamma, desc='FracLinPot'):
        """ Constructor """
        self.gamma = gamma
        self.desc = desc + ', ' + r'$\gamma={{{}}}$'.format(gamma)

    def phi(self, u):
        """ Returns phi(u), the value of the zero-potential at the points u"""
        return ( (u<1)*np.maximum(2 - u, 1)**(-self.gamma) +
                 (u>=1)*(1 + self.gamma*(u - 1)) )

    def phi_prime(self, u):
        """ Returns phi'(u), the first derivative of the zero-potential at the points u """
        return ( (u<1)*self.gamma*np.maximum(2 - u, 1)**(-(1+self.gamma)) +
                 (u>=1)*self.gamma )

    def phi_double_prime(self, u):
        """ Returns phi''(u), the second derivative of the zero-potential at the points u """
        return (u<1)*self.gamma*(1+self.gamma)*np.maximum(2 - u, 1)**(-(2+self.gamma))

    def phi_inv(self, u):
        """ Returns phi^{-1}(u), the inverse function of the zero-potential at the points u """
        return (u<1)*(2 - np.minimum(u,1)**(-1/self.gamma)) + (u>=1)*(1 + (u - 1)/self.gamma)

    def phi_inv_prime(self, u):
        """ Returns phi^{-1}'(u), the first derivative of the inverse
            function of the zero-potential at the points u """
        return 1/self.phi_prime(self.phi_inv(u))

    def isconvex(self):
        """ Returns True if phitilde(u) = max(phi(u), 0) is a convex function. """
        return True

    def bounds_asymp(self):
        """ Returns constants C and epsilon s.t. f_phi(x) <= C x**(1+epsilon)
            for all x >= 1. See Theorem 2 in paper for details. """
        return 1/self.gamma + (1-1/self.gamma), 1

    def l_psi(self):
        """ Returns the strong convexity constant l_psi and the norm (1-norm / total variation norm)
            of the Csizar divergence associated with the Linear Fractional Potential. """
        return 1/self.gamma, 1

    def gen_ccode(self):
        """ Generates a c-code snippet used for fast numerical integration """
        return ['   double z = -eta*(loss + nu);\n',
                '   if(z<1){{\n'.format(self.u0),
                '     return pow(2.0 - z, -{});}}\n'.format(self.gamma),
                '   else{\n',
                '     return 1.0 + {}*(z - 1.0);}}\n'.format(self.gamma),
                '   }']


class FractionalExponentialPotential(OmegaPotential):
    """ A fractional-exponential potential formed by stitching together a
        fractional and an exponential function """

    def __init__(self, gamma=2, desc='FracExpPot'):
        """ Constructor """
        self.gamma = gamma
        self.desc = desc + ', ' + r'$\gamma={{{}}}$'.format(gamma)

    def phi(self, u):
        """ Returns phi(u), the value of the zero-potential at the points u"""
        return ( (u<1)*np.maximum(2 - u, 1)**(-self.gamma) +
                 (u>=1)*np.exp(self.gamma*(u - 1)) )

    def phi_prime(self, u):
        """ Returns phi'(u), the first derivative of the zero-potential at the points u """
        return ( (u<1)*self.gamma*np.maximum(2 - u, 1)**(-(1+self.gamma)) +
                 (u>=1)*self.gamma*np.exp(self.gamma*(u - 1)) )

    def phi_double_prime(self, u):
        """ Returns phi''(u), the second derivative of the zero-potential at the points u """
        return ( (u<1)*self.gamma*(1+self.gamma)*np.maximum(2 - u, 1)**(-(2+self.gamma))
                 + (u>=1)*self.gamma**2*np.exp(self.gamma*(u - 1)) )

    def phi_inv(self, u):
        """ Returns phi^{-1}(u), the inverse function of the zero-potential at the points u """
        return (u<1)*(2 - np.minimum(u,1)**(-1/self.gamma)) + (u>=1)*(1 + np.log(u)/self.gamma)

    def phi_inv_prime(self, u):
        """ Returns phi^{-1}'(u), the first derivative of the inverse
            function of the zero-potential at the points u """
        return 1/self.phi_prime(self.phi_inv(u))

    def isconvex(self):
        """ Returns True if phitilde(u) = max(phi(u), 0) is a convex function. """
        return True

    def bounds_asymp(self):
        """ Returns constants C and epsilon s.t. f_phi(x) <= C x**(1+epsilon)
            for all x >= 1. See Theorem 2 in paper for details. """
        return 1/self.gamma + 1, 0

    def l_psi(self):
        """ Returns the strong convexity constant l_psi and the norm (1-norm / total variation norm)
            of the Csizar divergence associated with the Linear Fractional Potential. """
        return 1/self.gamma, 1

    def gen_ccode(self):
        """ Generates a c-code snippet used for fast numerical integration """
        return ['   double z = -eta*(loss + nu);\n',
                '   if(z<1){{\n',
                '     return pow(2.0 + - z, -{});}}\n'.format(self.gamma),
                '   else{\n',
                '     return exp({}*(z-1));}}\n'.format(self.gamma),
                '   }']


class ExpPPotential(OmegaPotential):
    """ A potential given by a composition of an exponential and a p norm """

    def __init__(self, p, gamma=1, desc='ExpPPot', **kwargs):
        """ Constructor """
        self.p, self.gamma = p, gamma
        self.desc = desc = desc + ', ' + r'$p={{{}}}, gamma={{{}}}$'.format(p, gamma)
        if kwargs.get('M') is not None:
            self.M = kwargs.get('M')

    def phi(self, u):
        """ Returns phi(u), the value of the ExpP-potential at the points u"""
        return ( (u<=0)*np.exp(self.gamma*u)
                 + (u>0)*(self.gamma*(self.p-1)*np.abs(u))**(1/(self.p-1)) )

    def phi_inv(self, u):
        """ Returns phi^{-1}(u), the inverse function of the zero-potential at the points u """
        return (u<=1)*np.log(u)/self.gamma + (u>1)*(u**(self.p-1)-1)/self.gamma/(self.p-1)

    def isconvex(self):
        """ Returns True if phitilde(u) = max(phi(u), 0) is a convex function. """
        return True

    def bounds_asymp(self):
        """ Returns constants C and epsilon s.t. f_phi(x) <= C x**(1+epsilon)
            for all x >= 1. See Theorem 2 in paper for details. """
        return 1/self.gamma/self.p/(self.p-1), self.p-1

    def l_psi(self):
        """ Returns the strong convexity constant l_psi and the norm (e.g. 1-norm / total variation norm)
            of the Csizar divergence associated with the Linear Fractional Potential. """
        return 1/self.gamma, 2/(3-self.p)

    def gen_ccode(self):
        """ Generates a c-code snippet used for fast numerical integration """
        return ['   double z = -eta*(loss + nu);\n',
                '   if(z>0){\n',
                '     return pow({}*(z + {}), {});}}\n'.format(self.gamma*(self.p - 1), 1/self.gamma/(self.p-1), 1/(self.p-1)),
                '   else{\n',
                '     return exp({}*z);}}\n'.format(self.gamma),
                '   }']

    def gen_KLccode(self):
        """ Generates a c-code snippet used for fast numerical integration for KL divergence"""
        return ['   double z = -eta*(loss + nu);\n',
                '   if(z>0){\n',
                '     return pow({0}*(z + {1}), {2})*log(pow({0}*(z + {1}), {2}));}}\n'.format(self.gamma*(self.p - 1), 1/self.gamma/(self.p-1), 1/(self.p-1)),
                '   else{\n',
                '     return {0}*z*exp({0}*z);}}\n'.format(self.gamma),
                '   }']



class pExpPotential(OmegaPotential):
    """ A potential given by a composition of a p-norm and
        an exponential potential """

    def __init__(self, p, gamma=1, desc='pExpPot', **kwargs):
        """ Constructor """
        self.p, self.gamma = p, gamma
        self.c_omega, self.d_omega = 1/gamma, 0
        self.desc = desc + ', ' + r'$p={{{}}}, gamma={{{}}}$'.format(p, gamma)
        if kwargs.get('M') is not None:
            self.M = kwargs.get('M')

    def phi(self, u):
        """ Returns phi(u), the value of the Pexp-potential at the points u"""
        return ( (u>=-1/self.gamma/(self.p-1))*(u<0)*(self.gamma*(self.p-1)*np.abs(u+1/self.gamma/(self.p-1)))**(1/(self.p-1)) +
                 (u>=0)*np.exp(self.gamma*u) )

    def phi_inv(self, u):
        """ Returns phi^{-1}(u), the inverse function of the zero-potential at the points u """
        return ((u<1)*(u**(self.p-1) - 1)/self.gamma/(self.p-1)
                + (u>=1)*np.log(u)/self.gamma)

    def isconvex(self):
        """ Returns True if phitilde(u) = max(phi(u), 0) is a convex function. """
        return True

    def gen_ccode(self):
        """ Generates a c-code snippet used for fast numerical integration """
        return ['   double z = -eta*(loss + nu);\n',
                '   if(z>0){\n',
                '     return exp({0}*z);}}\n'.format(self.gamma),
                '   else if(z>{}){{\n'.format(-1/self.gamma/(self.p-1)),
                '     return pow({0}*(z + {1}), {2});}}\n'.format(self.gamma*(self.p - 1), 1/self.gamma/(self.p-1), 1/(self.p-1)),
                '   else{\n',
                '     return 0.0;}\n',
                '   }']

    def gen_KLccode(self):
        """ Generates a c-code snippet used for fast numerical integration """
        return ['   double z = -eta*(loss + nu);\n',
                '   if(z>0){\n',
                '     return {0}*z*exp({0}*z);}}\n'.format(self.gamma),
                '   else if(z>{}){{\n'.format(-1/self.gamma/(self.p-1)),
                '     return pow({0}*(z + {1}), {2})*log(pow({0}*(z + {1}), {2}));}}\n'.format(self.gamma*(self.p - 1), 1/self.gamma/(self.p-1), 1/(self.p-1)),
                '   else{\n',
                '     return 0.0;}\n',
                '   }']
