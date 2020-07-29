from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mc

__all__ = ["scatter", "camera_scatter"]

# Some good camera angles for scatter plots.
camera_scatter = {
    'dtlz2': {'3d': [60,20], '4d':[-60,30], '8d': [22,21]}, \
    'dtlz2-nbi': {'3d': [60,20], '4d':[-60,30], '8d': [-60,30]}, \
    'debmdk': {'3d': [-30,15], '4d': [-20,32], '8d': [-60,30]}, \
    'debmdk-nbi': {'3d': [-60,30], '4d': [-60,30], '8d': [-60,30]}, \
    'debmdk-all': {'3d': [-60,30], '4d': [-60,30], '8d': [-60,30]}, \
    'debmdk-all-nbi': {'3d': [-60,30], '4d': [-60,30], '8d': [-60,30]}, \
    'dtlz8': {'3d': [-60,30], '4d': [-60,30], '6d': [-60,30], '8d': [-60,30]}, \
    'dtlz8-nbi': {'3d': [-60,30], '4d': [-60,30], '6d': [-60,30], '8d': [-60,30]}, \
    'c2dtlz2': {'3d': [45,15], '4d': [-20,40], '5d': [-25,30], '8d': [-25,30]}, \
    'c2dtlz2-nbi': {'3d': [45,15], '4d': [-20,40], '5d': [-25,30], '8d': [-25,30]}, \
    'cdebmdk': {'3d': [20,15], '4d': [-60,30], '8d': [-60,30]}, \
    'cdebmdk-nbi': {'3d': [20,15], '4d': [-60,30], '8d': [-60,30]}, \
    'c0dtlz2': {'3d': [20,25], '4d': [-60,30], '8d': [-60,30]}, \
    'c0dtlz2-nbi': {'3d': [20,25], '4d': [-60,30], '8d': [-60,30]}, \
    'crash-nbi': {'3d': [30,25]}, 'crash-c1-nbi': {'3d': [30,25]}, 'crash-c2-nbi': {'3d': [30,25]}, \
    'gaa': {'10d': [-60,30]}, \
    'gaa-nbi': {'10d': [-60,30]}
}

def scatter(A, plt = None, s = 1, c = mc.TABLEAU_COLORS['tab:blue'], **kwargs):
    
    # all other parameters
    # by default label_prefix is $f_n$
    label_prefix = kwargs['label_prefix'] if (kwargs is not None and 'label_prefix' in kwargs) \
                            else r"$f_{:d}$"
    # default label font size is 'large'
    label_fontsize = kwargs['label_fontsize'] if (kwargs is not None and 'label_fontsize' in kwargs) \
                            else 'large'
    # plot first 3 axes by default
    axes = kwargs['axes'] if (kwargs is not None and 'axes' in kwargs) else [0, 1, 2]
    # azimuth is -60 and elevation is 30 by default
    euler = kwargs['euler'] if (kwargs is not None and 'euler' in kwargs) else [-60, 30]
    # by default, take the entire range
    xbound = kwargs['xbound'] if (kwargs is not None and 'xbound' in kwargs) else None
    ybound = kwargs['ybound'] if (kwargs is not None and 'ybound' in kwargs) else None
    zbound = kwargs['zbound'] if (kwargs is not None and 'zbound' in kwargs) else None
    # by default, no title
    title = kwargs['title'] if (kwargs is not None and 'title' in kwargs) else None

    if plt is not None:
        fig = plt.figure()
        if title is not None:
            fig.suptitle(title)
        if A.shape[1] < 3:
            ax = fig.gca()
            ax.scatter(A[:,axes[0]], A[:,axes[1]], s = s, c = c)
            ax.set_xbound(ax.get_xbound() if xbound is None else xbound)
            ax.set_ybound(ax.get_ybound() if ybound is None else ybound)
            ax.set_xlabel(label_prefix.format(axes[0] + 1), fontsize = label_fontsize)
            ax.set_ylabel(label_prefix.format(axes[1] + 1), fontsize = label_fontsize)
        else:
            ax = Axes3D(fig)
            ax.scatter(A[:,axes[0]], A[:,axes[1]], A[:,axes[2]], s = s, c = c) 
            ax.set_xbound(ax.get_xbound() if xbound is None else xbound)
            ax.set_ybound(ax.get_ybound() if ybound is None else ybound)
            ax.set_zbound(ax.get_zbound() if zbound is None else zbound)
            ax.set_xlabel(label_prefix.format(axes[0] + 1), fontsize = label_fontsize)
            ax.set_ylabel(label_prefix.format(axes[1] + 1), fontsize = label_fontsize)
            ax.set_zlabel(label_prefix.format(axes[2] + 1), fontsize = label_fontsize)
            ax.xaxis.set_rotate_label(False)
            ax.yaxis.set_rotate_label(False)
            ax.zaxis.set_rotate_label(False)
            ax.view_init(euler[1], euler[0])
        return fig,ax
    else:
        raise TypeError("`plt` can't be None, a valid pyplot object must be provided.") 
