'''
Collection of utility functions

@date May 25, 2015
'''

import numpy as np
import pickle, os
from matplotlib import pyplot as plt
from matplotlib import cm
import mpl_toolkits.mplot3d.axes3d as p3
from .Domains import nBox, DifferenceOfnBoxes


def plot_results(results, offset=500, directory=None, show=True):
        """ Plots and shows or saves (or both) the simulation results """
        # set up figures
        ylimits = [[np.Infinity, -np.Infinity] for i in range(3)]
        plt.figure(1)
        plt.title('cumulative regret, {} losses'.format(results[0].problem.lossfuncs[0].desc))
        plt.xlabel('t')
        plt.figure(2)
        plt.title('time-avg. cumulative regret, {} losses'.format(results[0].problem.lossfuncs[0].desc))
        plt.xlabel('t')
        plt.figure(3)
        plt.title(r'log time-avg. cumulative regret, {} losses'.format(results[0].problem.lossfuncs[0].desc))
        plt.xlabel('t')
        # and now plot, depending on what data is there
        for result in results:
            if result.algo in ['DA', 'OGD']:
                try:
                    plt.figure(1)
                    lavg = plt.plot(result.regs_norate['savg'][0], linewidth=2.0, label=result.label, rasterized=True)
                    plt.fill_between(np.arange(result.problem.T), result.regs_norate['perc_10'][0],
                                     result.regs_norate['perc_90'][0], color=lavg[0].get_color(), alpha=0.1, rasterized=True)
                    plt.figure(2)
                    ltavg = plt.plot(np.arange(result.problem.T)[offset:], result.regs_norate['tsavg'][0][offset:],
                                     linewidth=2.0, label=result.label, rasterized=True)
                    plt.fill_between(np.arange(offset,result.problem.T), result.regs_norate['tavg_perc_10'][0][offset:],
                                     result.regs_norate['tavg_perc_90'][0][offset:], color=ltavg[0].get_color(), alpha=0.1, rasterized=True)
                    plt.xlim((0, result.problem.T))
                    plt.figure(3)
                    lltsavg = plt.plot(np.arange(1,result.problem.T+1), result.regs_norate['tsavg'][0], linewidth=2.0,
                                       label=result.label, rasterized=True)
                    plt.fill_between(np.arange(1,result.problem.T+1), result.regs_norate['tavg_perc_10'][0],
                                    result.regs_norate['tavg_perc_90'][0], color=lltsavg[0].get_color(), alpha=0.1, rasterized=True)
                    plt.plot(np.arange(1,result.problem.T+1), result.regs_norate['tsavgbnd'][0], '--',
                             color=lltsavg[0].get_color(), linewidth=2, rasterized=True)
                    ylimits[2][0] = np.minimum(ylimits[2][0], np.min(result.regs_norate['tsavg'][0]))
                    ylimits[2][1] = np.maximum(ylimits[2][1], 1.1*np.max(result.regs_norate['tsavgbnd'][0]))
                except AttributeError: pass
                try:
                    for i,(T,eta) in enumerate(result.etaopts.items()):
                        plt.figure(1)
                        lavg = plt.plot(result.regs_etaopts['savg'][i][0:T], linewidth=2.0,
                                        label=result.label+' '+r' $\eta_{{opt}}(T={0:.1e}) = {1:.3f}$'.format(T, eta), rasterized=True)
                        plt.plot(np.arange(T,result.problem.T), result.regs_etaopts['savg'][i][T:], '--',
                                 color=lavg[0].get_color(), linewidth=2, rasterized=True)
                        plt.fill_between(np.arange(result.problem.T), result.regs_etaopts['perc_10'][i],
                                         result.regs_etaopts['perc_90'][i], color=lavg[0].get_color(), alpha=0.1, rasterized=True)
                        plt.figure(2)
                        ltavg = plt.plot(np.arange(offset,T), result.regs_etaopts['tsavg'][i][offset:T], linewidth=2.0,
                                         label=result.label+' '+r' $\eta_{{opt}}(T={0:.1e}) = {1:.3f}$'.format(T, eta), rasterized=True)
                        plt.plot(np.arange(T,result.problem.T), result.regs_etaopts['tsavg'][i][T:], '--',
                                 color=ltavg[0].get_color(), linewidth=2, rasterized=True)
                        plt.fill_between(np.arange(offset,result.problem.T), result.regs_etaopts['tavg_perc_10'][i][offset:],
                                         result.regs_etaopts['tavg_perc_90'][i][offset:], color=ltavg[0].get_color(), alpha=0.1, rasterized=True)
                        plt.xlim((0, result.problem.T))
                        plt.figure(3)
                        llogtavg = plt.plot(np.arange(1,result.problem.T+1), result.regs_etaopts['tsavg'][i],
                                            linewidth=2.0, label=result.label+' '+r' $\eta_{{opt}}(T={0:.1e}) = {1:.3f}$'.format(T, eta), rasterized=True)
                        plt.fill_between(np.arange(1,result.problem.T+1), result.regs_etaopts['tavg_perc_10'][i],
                                        result.regs_etaopts['tavg_perc_90'][i], color=llogtavg[0].get_color(), alpha=0.1, rasterized=True)
                        plt.plot(np.arange(1,result.problem.T+1), result.regs_etaopts['tsavgbnd'][i], '--',
                                 color=llogtavg[0].get_color(), linewidth=2, rasterized=True)
                        ylimits[2][0] = np.minimum(ylimits[2][0], np.min(result.regs_etaopts['tsavg'][0]))
                        ylimits[2][1] = np.maximum(ylimits[2][1], 1.1*np.max(result.regs_etaopts['tsavgbnd'][0]))
    #
                except AttributeError: pass
                try:
                    for i,eta in enumerate(result.etas):
                        plt.figure(1)
                        lavg = plt.plot(result.regs_etas['savg'][i], linewidth=2.0, label=result.label+' '+r' $\eta = {0:.3f}$'.format(eta), rasterized=True)
                        plt.fill_between(np.arange(result.problem.T), result.regs_etas['perc_10'][i],
                                         result.regs_etas['perc_90'][i], color=lavg[0].get_color(), alpha=0.1, rasterized=True)
                        plt.figure(2)
                        ltavg = plt.plot(np.arange(offset,result.problem.T), result.regs_etas['tsavg'][i][offset:],
                                         linewidth=2.0, label=result.label+' '+r'$\eta = {0:.3f}$'.format(eta), rasterized=True)
                        plt.fill_between(np.arange(offset,result.problem.T), result.regs_etas['tavg_perc_10'][i][offset:],
                                         result.regs_etas['tavg_perc_90'][i][offset:], color=ltavg[0].get_color(), alpha=0.1, rasterized=True)
                        plt.xlim((0, result.problem.T))
                        plt.figure(3)
                        llogtavg = plt.plot(np.arange(1,result.problem.T+1), result.regs_etas['tsavg'][i], linewidth=2.0,
                                            label=result.label+' '+r' $\eta = {0:.3f}$'.format(eta), rasterized=True)
                        plt.fill_between(np.arange(1,result.problem.T+1), result.regs_etas['tavg_perc_10'][i],
                                         result.regs_etas['tavg_perc_90'][i], color=llogtavg[0].get_color(), alpha=0.1, rasterized=True)
                        plt.plot(np.arange(1,result.problem.T+1), result.regs_etas['tsavgbnd'][i], '--',
                                 color=llogtavg[0].get_color(), linewidth=2, rasterized=True)
                        ylimits[2][0] = np.minimum(ylimits[2][0], np.min(result.regs_etaos['tsavg'][0]))
                        ylimits[2][1] = np.maximum(ylimits[2][1], 1.1*np.max(result.regs_etas['tsavgbnd'][0]))
                except AttributeError: pass
                try:
                    for i,alpha in enumerate(result.alphas):
                        plt.figure(1)
                        lavg = plt.plot(result.regs_alphas['savg'][i], linewidth=2.0,
                                 label=result.label+' '+r' $\eta_t = {0} \cdot t^{{{1}}}$'.format(result.thetas[i], -alpha), rasterized=True)
                        plt.fill_between(np.arange(result.problem.T), result.regs_alphas['perc_10'][i],
                                         result.regs_alphas['perc_90'][i], color=lavg[0].get_color(), alpha=0.1, rasterized=True)
                        plt.figure(2)
                        ltavg = plt.plot(np.arange(result.problem.T)[offset:], result.regs_alphas['tsavg'][i][offset:], linewidth=2.0,
                                 label=result.label+' '+r' $\eta_t = {0} \cdot t^{{{1}}}$'.format(result.thetas[i], -alpha), rasterized=True)
                        plt.fill_between(np.arange(offset,result.problem.T), result.regs_alphas['tavg_perc_10'][i][offset:],
                                         result.regs_alphas['tavg_perc_90'][i][offset:], color=ltavg[0].get_color(), alpha=0.1, rasterized=True)
                        plt.xlim((0, result.problem.T))
                        plt.figure(3)
                        lltsavg = plt.plot(np.arange(1,result.problem.T+1), result.regs_alphas['tsavg'][i], linewidth=2.0,
                                             label=result.label+' '+r' $\eta_t = {0} \cdot t^{{{1}}}$'.format(result.thetas[i], -alpha), rasterized=True)
                        plt.fill_between(np.arange(1,result.problem.T+1), result.regs_alphas['tavg_perc_10'][i],
                                        result.regs_alphas['tavg_perc_90'][i], color=lltsavg[0].get_color(), alpha=0.1, rasterized=True)
                        plt.plot(np.arange(1,result.problem.T+1), result.regs_alphas['tsavgbnd'][i], '--', color=lltsavg[0].get_color(),
                                 linewidth=2.0, rasterized=True)
                        ylimits[2][0] = np.minimum(ylimits[2][0], np.min(result.regs_alphas['tsavg'][0]))
                        ylimits[2][1] = np.maximum(ylimits[2][1], 1.1*np.max(result.regs_alphas['tsavgbnd'][0]))
                except AttributeError: pass
            else:
                plt.figure(1)
                lavg = plt.plot(result.regs_norate['savg'][0], linewidth=2.0, label=result.label, rasterized=True)
                plt.fill_between(np.arange(result.problem.T), result.regs_norate['perc_10'][0],
                                 result.regs_norate['perc_90'][0], color=lavg[0].get_color(), alpha=0.1, rasterized=True)
                plt.figure(2)
                ltavg = plt.plot(np.arange(result.problem.T)[offset:], result.regs_norate['tsavg'][0][offset:],
                                 linewidth=2.0, label=result.label, rasterized=True)
                plt.fill_between(np.arange(offset,result.problem.T), result.regs_norate['tavg_perc_10'][0][offset:],
                                 result.regs_norate['tavg_perc_90'][0][offset:], color=ltavg[0].get_color(), alpha=0.1, rasterized=True)
                plt.xlim((0, result.problem.T))
                plt.figure(3)
                lltsavg = plt.plot(np.arange(1,result.problem.T+1), result.regs_norate['tsavg'][0], linewidth=2.0,
                                   label=result.label, rasterized=True)
                plt.fill_between(np.arange(1,result.problem.T+1), result.regs_norate['tavg_perc_10'][0],
                                result.regs_norate['tavg_perc_90'][0], color=lltsavg[0].get_color(), alpha=0.1, rasterized=True)
                plt.plot(np.arange(1,result.problem.T+1), result.regs_norate['tsavgbnd'][0], '--',
                         color=lltsavg[0].get_color(), linewidth=2, rasterized=True)
                ylimits[2][0] = np.minimum(ylimits[2][0], np.min(result.regs_norate['tsavg'][0]))
                ylimits[2][1] = np.maximum(ylimits[2][1], 1.1*np.max(result.regs_norate['tsavgbnd'][0]))

        # make plots pretty and show legend
        plt.figure(1)
        plt.legend(loc='upper left', prop={'size':13}, frameon=False)
        plt.figure(2)
        plt.legend(loc='upper right', prop={'size':13}, frameon=False)
        plt.figure(3)
        plt.yscale('log'), plt.xscale('log')
#         plt.ylim(np.log(ylimits[2][0]), np.log(ylimits[2][1]))
        plt.legend(loc='upper right', prop={'size':13}, frameon=False)
        if directory:
            os.makedirs(directory+'figures/', exist_ok=True) # this could probably use a safer implementation
            filename = '{}{}_{}_'.format(directory+'figures/', results[0].problem.desc, results[0].problem.lossfuncs[0].desc)
            plt.figure(1)
            plt.savefig(filename + 'cumloss.pdf', bbox_inches='tight', dpi=300)
            plt.figure(2)
            plt.savefig(filename + 'tavgloss.pdf', bbox_inches='tight', dpi=300)
            plt.figure(3)
            plt.savefig(filename + 'loglogtavgloss.pdf', bbox_inches='tight', dpi=300)
        if show:
            plt.show()


def plot_dims(results, directory=None, show=True):
        """ Plots and shows or saves (or both) the simulation results """
        # set up figures
#         ylimits = [np.Infinity, -np.Infinity]
        f = plt.figure()
        plt.title(r'log time-avg. cumulative regret, {} losses'.format(results[0].problem.lossfuncs[0].desc))
        plt.xlabel('t')
        dim_styles = {2:'--', 3:'-.', 4:':'}
        # and now plot, depending on what data is there
        for loss_results in results:
            for result in loss_results:
                lltsavg = plt.plot(np.arange(1,result.problem.T+1), result.regs_norate['tsavg'], linewidth=2.0,
                                   linestyle=dim_styles[result.dim], label=result.label, rasterized=True)
                plt.fill_between(np.arange(1,result.problem.T+1), result.regs_norate['tavg_perc_10'], result.regs_norate['tavg_perc_90'],
                                 linestyle=dim_styles[result.dim], color=lltsavg[0].get_color(), alpha=0.1, rasterized=True)
        # make plots pretty and show legend
        plt.yscale('log'), plt.xscale('log')
        plt.legend(loc='upper right', prop={'size':13}, frameon=False)
        if directory:
            os.makedirs(directory+'figures/', exist_ok=True) # this could probably use a safer implementation
            filename = '{}{}_{}_'.format(directory+'figures/', results[0].problem.desc, results[0].problem.lossfuncs[0].desc)
            plt.savefig(filename + 'loglogtavgloss.pdf', bbox_inches='tight', dpi=300)
        if show:
            plt.show()
        plt.close()



def plot_loglogs(results, directory=None, show=True, bounds=True, **kwargs):
        """ Plots and shows or saves (or both) the simulation results """
        # set up figures
        f = plt.figure()
        loss_title = list(results[0].values())[0].problem.lossfuncs[0].desc
        plt.title(r'log time-avg. cumulative regret, {} losses'.format(loss_title))
        plt.xlabel('t')
        colors = ['k', 'r', 'g', 'b', 'c', 'm', 'y']*3
        loss_styles = ['-', '--', '-.', ':']*3
        labs = kwargs.get('labels')
        # and now plot, depending on what data is there
        for i,loss_results in enumerate(results):
            for j,key in enumerate(loss_results.keys()):
                r = loss_results[key]
                if labs is not None:
                    lab = labs[i][j]
                    print(lab)
                else:
                    lab = r.label
                lltsavg = plt.plot(np.arange(1,r.problem.T+1), r.regs_norate['tsavg'][0], linewidth=2.0,
                                   linestyle=loss_styles[i], color=colors[j], label=lab, rasterized=True)
                plt.fill_between(np.arange(1,r.problem.T+1), r.regs_norate['tavg_perc_10'][0], r.regs_norate['tavg_perc_90'][0],
                                 linestyle=loss_styles[i], color=colors[j], alpha=0.1, rasterized=True)
                if bounds:
                    try:
                        plt.plot(np.arange(1,r.problem.T+1), r.regs_norate['tsavgbnd'][0],
                                 color=colors[j], linewidth=3, rasterized=True)
                    except IndexError: pass
        # make plots pretty and show legend
        plt.yscale('log'), plt.xscale('log')
        plt.legend(prop={'size':12}, frameon=False, **kwargs)  #loc='lower center',
        if directory:
            os.makedirs(directory, exist_ok=True) # this could probably use a safer implementation
            filename = '{}{}_{}_'.format(directory, list(results[0].values())[0].problem.desc,
                                         list(results[0].values())[0].problem.lossfuncs[0].desc)
            plt.savefig(filename + 'loglogtavgloss.pdf', bbox_inches='tight', dpi=300)
        if show:
            plt.show()
        plt.close()


def plot_snapshots(results, times, filename=None, show=False, **kwargs):
    """ Creates a sequence of plots from the pltdata array in the results at the
        time steps specified in times (will be ordered increasing).
        Here results is an iterable of results. The resulting figure will have
        len(results) x len(times) plots. """
    pltpoints = results[0].problem.pltpoints
    fig = plt.figure(figsize=kwargs.get('figsize'))
    # idk why the FUCK this does not work just using np arrays!?
    zmax = np.max([np.max([np.max([np.max(df) for df in dflat]) for dflat in result.pltdata]) for result in results])
    zmin = np.min([np.min([np.min([np.min(df) for df in dflat]) for dflat in result.pltdata]) for result in results])
    for i,result in enumerate(results):
        bbox = result.problem.domain.bbox()
        for j,time in enumerate(np.sort(times)):
            ax = fig.add_subplot(len(results), len(times), len(times)*i+j+1, projection='3d')
            for points,dat in zip(pltpoints, result.pltdata[time]):
                ax.plot_trisurf(points[:,0], points[:,1], dat, cmap=plt.get_cmap('jet'),
                               linewidth=0, vmin=zmin, vmax=zmax)
            # Setting the axes properties
            ax.set_xlim3d(bbox.bounds[0])
            ax.set_xlabel('$s_1$')
            ax.set_ylim3d(bbox.bounds[1])
            ax.set_ylabel('$s_2$')
            ax.set_zlim3d([-0.1, zmax])
            ax.set_zlabel('$x$')
            ax.set_title('t={}'.format(time))
            ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.view_init(elev=kwargs.get('elev'), azim=kwargs.get('azim'))
    plt.tight_layout()
#     if directory:
#         os.makedirs(directory, exist_ok=True) # this could probably use a safer implementation
#         filename = '{}{}_{}_'.format(directory, results[0].problem.desc,
#                                      results[0].problem.lossfuncs[0].desc)
#         plt.savefig(filename + 'snapshots.pdf', bbox_inches='tight', dpi=300)
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight', dpi=300)
    if show:
        plt.show()
    plt.close()


def save_results(results, directory):
    """ Serializes a results object for persistent storage using the pickle module. """
    os.makedirs(directory, exist_ok=True) # this could probably use a safer implementation
    slope_txt = []
    for result in results:
        try:
            [slope_txt.append('{}, Empirical: {}\n'.format(result.label, val[0])) for val in result.slopes.values()]
            [slope_txt.append('{}, Bounds: {}\n'.format(result.label, val[0])) for val in result.slopes_bnd.values()]
            del result.problem.pltpoints, result.problem.data
        except (AttributeError, IndexError):
            pass
    slopes_name = '{}{}_{}_slopes.txt'.format(directory, results[0].problem.desc,
                                       results[0].problem.lossfuncs[0].desc)
    with open(slopes_name, 'w') as f:
        f.writelines(slope_txt)
    pigglname = '{}{}_{}.piggl'.format(directory, results[0].problem.desc,
                                       results[0].problem.lossfuncs[0].desc)
    with open(pigglname, 'wb') as f:
        pickle.dump(results, f, pickle.HIGHEST_PROTOCOL)


def visualize_potentials(potentials, xlim=(-1,5), **kwargs):
    u = np.linspace(xlim[0], xlim[1], 1000)
    plt.figure(figsize=kwargs.get('figsize'))
    labels = kwargs.get('labels')
    if labels is None:
        labels = [pot.desc for pot in potentials]
    if kwargs.get('semilogy') == True:
        for vals,label in zip([pot.phi(u) for pot in potentials], labels):
            plt.semilogy(u, 1+vals, label=label)
    else:
        for vals,label in zip([pot.phi(u) for pot in potentials], labels):
            plt.plot(u, vals, label=label, linewidth=2)
    plt.ylim(kwargs.get('ylim'))
    plt.xlabel('$u$', fontsize=15)
    plt.ylabel('$\phi(u)$', fontsize=15)
    plt.legend(loc=kwargs.get('loc'), frameon=False)
    plt.title('Various $\omega$-potentials')
    plt.tight_layout()
    if kwargs.get('filename') is not None:
        plt.savefig(kwargs.get('filename'), bbox_inches='tight', dpi=300)
    if kwargs.get('show') is not False:
        plt.show()
    plt.close()


def circular_tour(domain, N):
    """ Returns a sequence of N points that wander around in a circle
        in the domain. Used for understanding various learning rates. """
    if domain.n != 2:
        raise Exception('For now circular_tour only works in dimension 2')
    if isinstance(domain, nBox):
        center = np.array([0.5*(bnd[0]+bnd[1]) for bnd in domain.bounds])
        halfaxes = np.array([0.75*0.5*(bnd[1]-bnd[0]) for bnd in domain.bounds])
        return np.array([center[0] + halfaxes[0]*np.cos(np.linspace(0,2*np.pi,N)),
                         center[1] + halfaxes[1]*np.sin(np.linspace(0,2*np.pi,N))]).T
    if isinstance(domain, DifferenceOfnBoxes) and (len(domain.inner) == 1):
        lengths = [bound[1] - bound[0] for bound in domain.outer.bounds]
        weights = np.array(lengths*2)/2/np.sum(lengths)
        bnds_inner, bnds_outer = domain.inner[0].bounds, domain.outer.bounds
        xs = np.concatenate([np.linspace(0.5*(bnds_inner[0][0]+bnds_outer[0][0]), 0.5*(bnds_inner[0][1]+bnds_outer[0][1]), weights[0]*N),
                             0.5*(bnds_outer[0][1]+bnds_inner[0][1])*np.ones(weights[1]*N),
                             np.linspace(0.5*(bnds_inner[0][1]+bnds_outer[0][1]), 0.5*(bnds_inner[0][0]+bnds_outer[0][0]), weights[2]*N),
                             0.5*(bnds_outer[0][0]+bnds_inner[0][0])*np.ones(weights[3]*N)])
        ys = np.concatenate([0.5*(bnds_outer[1][0]+bnds_inner[1][0])*np.ones(weights[0]*N),
                             np.linspace(0.5*(bnds_outer[1][0]+bnds_inner[1][0]), 0.5*(bnds_inner[1][1]+bnds_outer[1][1]), weights[1]*N),
                             0.5*(bnds_outer[1][1]+bnds_inner[1][1])*np.ones(weights[2]*N),
                             np.linspace(0.5*(bnds_inner[1][1]+bnds_outer[1][1]), 0.5*(bnds_inner[1][0]+bnds_outer[1][0]), weights[3]*N)])
        return np.array([xs, ys]).T
    else:
        raise Exception('For now circular_tour only works on nBoxes and the difference of 2 nBoxes')

def quicksample(bounds, A, eta):
    """ Function returning actions sampled from the solution of the Dual Averaging
        update on an Box with Affine losses, Exponential Potential. """
    C1, C2 = np.exp(-eta*A*bounds[:,0]), np.exp(-eta*A*bounds[:,1])
    Finv = lambda U: -np.log(C1 - (C1-C2)*U)/A/eta
    np.random.seed()
    return Finv(np.random.rand(*A.shape))

def CNR_worker(prob, *args, **kwargs):
    """ Helper function for wrapping class methods to allow for easy
        use of the multiprocessing package for parallel computing """
    return prob.run_simulation(*args, **kwargs)
