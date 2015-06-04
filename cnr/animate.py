'''
Some helper functions to produce fancy animations

@date: May 8, 2015
'''

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation


def save_animations(results, length=10, directory=None, show=False, **kwargs):
    """ Takes in a list of Result objects and creates and saves an animation of the
        evolution of the pdf over time of total duration length seconds. """
    # save all result animations (for now)
    for r, result in enumerate(results):
        T = result.problem.T
        frames = T-1
        interval = length/T*1000
        try:
            pltdata = result.pltdata
            pltpoints = result.problem.pltpoints

            # Creating figure, attaching 3D axis to the figure
            fig = plt.figure()
            ax = p3.Axes3D(fig)

            # Extract some information for the plots
            bbox = result.problem.domain.bbox()
            # idk why the FUCK this does not work just using np arrays!?
            zmax = np.max([np.max([np.max(df) for df in dflat]) for dflat in pltdata])
            zmin = np.min([np.min([np.min(df) for df in dflat]) for dflat in pltdata])

            # create initial object
            for points,dat in zip(pltpoints, pltdata[0]):
#                 print(points, dat, pltpoints, pltdata)
                plot = ax.plot_trisurf(points[:,0], points[:,1], dat, cmap=plt.get_cmap('jet'), vmin=zmin, vmax=zmax)
            # Setting the axes properties
            ax.set_xlim3d(bbox.bounds[0])
            ax.set_xlabel('$s_1$')
            ax.set_ylim3d(bbox.bounds[1])
            ax.set_ylabel('$s_2$')
            ax.set_zlim3d([-0.5, zmax])
            ax.set_zlabel('$x$')
            ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.view_init(elev=kwargs.get('elev'), azim=kwargs.get('azim'))

            def update_plot(framenum, data, plot):
                ax.clear()
                for points,dat in zip(pltpoints, data[framenum]):
                    plot = ax.plot_trisurf(points[:,0], points[:,1], dat, linewidth=0, cmap=plt.get_cmap('jet'), vmin=zmin, vmax=zmax)
                ax.set_xlim3d(bbox.bounds[0])
                ax.set_xlabel('$s_1$')
                ax.set_ylim3d(bbox.bounds[1])
                ax.set_ylabel('$s_2$')
                ax.set_zlim3d([-0.5, zmax])
                ax.set_zlabel('$x$')
                ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                ax.view_init(elev=kwargs.get('elev'), azim=kwargs.get('azim'))
                return plot

            # Creating the Animation object
            pdf_ani = animation.FuncAnimation(fig, update_plot, frames, fargs=(pltdata, plot),
                                              interval=interval, blit=False)
            if directory is not None:
                pdf_ani.save('{}animation_{}.mp4'.format(directory, r),
                             extra_args=['-vcodec', 'libx264'])
            if show:
                plt.show()
        except AttributeError: pass



def save_animations_NIPS2(res1, res2, length=10, filename=None, show=False, **kwargs):
    """ Takes in a list of Result objects and creates and saves an animation of the
        evolution of the pdf over time of total duration length seconds. """
    # save all result animations (for now)
    T = res1.problem.T
    frames = 60*length
    interval = 1/60*1000
    pltpoints1, pltpoints2 = res1.problem.pltpoints, res2.problem.pltpoints
    idcs = [int(t/frames*T) for t in range(frames)]
    pltdata1 = np.array([res1.pltdata[i] for i in idcs])
    pltdata2 = np.array([res2.pltdata[i] for i in idcs])

    # Creating figure, attaching 3D axis to the figure
    fig = plt.figure(figsize=kwargs.get('figsize'))
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')

    # Extract some information for the plots
    bbox1, bbox2 = res1.problem.domain.bbox(), res2.problem.domain.bbox()
    # idk why the FUCK this does not work just using np arrays!?
    zmax1 = np.max([np.max([np.max(df) for df in dflat]) for dflat in pltdata1])
    zmax2 = np.max([np.max([np.max(df) for df in dflat]) for dflat in pltdata2])
    zmin1 = np.min([np.min([np.min(df) for df in dflat]) for dflat in pltdata1])
    zmin2 = np.min([np.min([np.min(df) for df in dflat]) for dflat in pltdata2])

    # create initial objects
    for points,dat in zip(pltpoints1, pltdata1[0]):
        plot1 = ax1.plot_trisurf(points[:,0], points[:,1], dat, cmap=plt.get_cmap('jet'), vmin=zmin1, vmax=zmax1)
        ax1.set_title(kwargs.get('titles')[0])
    for points,dat in zip(pltpoints2, pltdata2[0]):
        plot2 = ax2.plot_trisurf(points[:,0], points[:,1], dat, cmap=plt.get_cmap('jet'), vmin=zmin2, vmax=zmax2)
        ax2.set_title(kwargs.get('titles')[1])

    for ax,bbox,zmax in zip([ax1,ax2], [bbox1,bbox2], [zmax1,zmax2]):
        # Setting the axes properties
        ax.set_xlim3d(bbox.bounds[0])
        ax.set_xlabel('$s_1$')
        ax.set_ylim3d(bbox.bounds[1])
        ax.set_ylabel('$s_2$')
        ax.set_zlabel('$x(s)$')
        ax.tick_params(labelsize=8)
        ax.set_zlim3d([-0.5, zmax])
        ax.set_zlim3d([-0.5, zmax])
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.view_init(elev=kwargs.get('elev'), azim=kwargs.get('azim'))
    plt.tight_layout()

    def update_plots(framenum, data, plot):
        ax1.clear(), ax2.clear()
        for points,dat in zip(pltpoints1, data[0][framenum]):
            plot1 = ax1.plot_trisurf(points[:,0], points[:,1], dat, linewidth=0, cmap=plt.get_cmap('jet'), vmin=zmin1, vmax=zmax1)
            ax1.set_title(kwargs.get('titles')[0])
        for points,dat in zip(pltpoints2, data[1][framenum]):
            plot2 = ax2.plot_trisurf(points[:,0], points[:,1], dat, linewidth=0, cmap=plt.get_cmap('jet'), vmin=zmin2, vmax=zmax2)
            ax2.set_title(kwargs.get('titles')[1])

        for ax,bbox,zmax in zip([ax1,ax2], [bbox1,bbox2], [zmax1,zmax2]):
            # Setting the axes properties
            ax.set_xlim3d(bbox.bounds[0])
            ax.set_xlabel('$s_1$')
            ax.set_ylim3d(bbox.bounds[1])
            ax.set_ylabel('$s_2$')
            ax.set_zlabel('$x(s)$')
            ax.tick_params(labelsize=8)
            ax.set_zlim3d([-0.5, zmax])
            ax.set_zlim3d([-0.5, zmax])
            ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.view_init(elev=kwargs.get('elev'), azim=kwargs.get('azim'))
        plt.tight_layout()
        return [plot1, plot2]

    # Creating the Animation object
    pdf_ani = animation.FuncAnimation(fig, update_plots, frames,
                                      fargs=([pltdata1, pltdata2], [plot1, plot2]),
                                      interval=interval, blit=False, )
    if filename is not None:
        pdf_ani.save(filename, extra_args=['-vcodec', 'libx264'])
    if show:
        plt.show()
