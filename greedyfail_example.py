'''
Comparison of Continuous No-Regret Algorithms:
When Greedy fails

@date: May 27, 2015
'''

# Set up infrastructure and basic problem parameters
import matplotlib as mpl
mpl.use('Agg') # this is needed when running on a linux server over terminal
import multiprocessing as mp
import numpy as np
import datetime, os
from ContNoRegret.Domains import unitbox
from ContNoRegret.LossFunctions import AffineLossFunction
from ContNoRegret.NoRegretAlgos import ContNoRegretProblem
from ContNoRegret.utils import CNR_worker, plot_results, save_results, circular_tour
from ContNoRegret.animate import save_animations
from ContNoRegret.Potentials import (ExponentialPotential, pNormPotential,
                                     ExpPPotential, pExpPotential)

# this is the location of the folder for the results
results_path = ''
desc = 'NIPS2_CNR_greedyfail'
tmpfolder = '/Volumes/tmp/' # if possible, choose this to be a RamDisk

# some flags for keeping a record of the simulation parameters
save_res = True
show_plots = False
save_anims = False
show_anims = False

T = 200 # Time horizon
L = 5.0 # Uniform bound on the Lipschitz constant
N = 2500 # Number of parallel algorithm instances
Ngrid = 500000 # Number of gridpoints for the sampling step
dom = unitbox(2)

# before running the computation, read this file so we can later
# save a copy in the results folder
with open(__file__, 'r') as f:
    thisfile = f.read()

# create a sequence of losses that is really baaaad for greedy
lossfuncs = [AffineLossFunction(dom, [L/2, 0], 0.25*L)] + [AffineLossFunction(dom, [(-1)**t*L, 0], 0.5*L) for t in np.arange(1,T)]
M = L
M4 = np.max([lossfunc.norm(4, tmpfolder=tmpfolder) for lossfunc in lossfuncs])
M83 = np.max([lossfunc.norm(8/3, tmpfolder=tmpfolder) for lossfunc in lossfuncs])

prob = ContNoRegretProblem(dom, lossfuncs, L, M, desc=desc)

# the following runs fine if the script is the __main__ method, but crashes when running from ipython
pool = mp.Pool(processes=mp.cpu_count()-1)
processes = []

processes.append(pool.apply_async(CNR_worker, (prob,N,'Greedy'),
                                  {'Ngrid':Ngrid, 'pid':len(processes),
                                   'tmpfolder':tmpfolder, 'label':'Greedy'}))

potentials = [ExponentialPotential(), pNormPotential(1.001), pNormPotential(1.01),
              pNormPotential(1.05), pNormPotential(1.75, M=M83)]

for pot in potentials:
    processes.append(pool.apply_async(CNR_worker, (prob,N,'DA'), {'opt_rate':True, 'Ngrid':Ngrid,
				  'potential':pot, 'pid':len(processes), 'tmpfolder':tmpfolder, 'label':pot.desc,
                  'results_path':results_path, 'KL':[]}))

# wait for the processes to finish an collect the results
results = [process.get() for process in processes]

# plot results and/or save a persistent copy (pickled) of the detailed results
timenow = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')# create a time stamp for unambiguously naming the results folder
results_directory = '{}{}/'.format(results_path, timenow)

if save_res:
    os.makedirs(results_directory, exist_ok=True) # this could probably use a safer implementation
#     plot_results(results, 100, results_directory, show_plots)
    if save_anims:
        save_animations(results, 10, results_directory, show_anims)
    save_results(results, results_directory)
    # store the previously read-in contents of this file in the results folder
    with open(results_directory+str(__file__), 'w') as f:
        f.write(thisfile)
else:
    plot_results(results, offset=100)
