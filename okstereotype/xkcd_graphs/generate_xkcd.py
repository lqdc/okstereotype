#!/usr/bin/env python
'''
@file plot_stuff.py
@date Wed 23 Jan 2013 02:14:07 PM EST
@author Roman Sinayev
@email roman.sinayev@gmail.com
@detail
'''
from xkcd_generator import XKCDify
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.dates import DateFormatter
# import pylab

def find_nearest_idx(array,value):
    return np.abs(array-value).argmin()

def index_to_xloc(index):
    return (index+3)*1.4/29

def plot_essay_len(l):
    np.random.seed(12)
    l = l/1000.
    y = np.array([42042, 37808, 34124, 29091, 23452, 18980, 15201, 11876, 9578, 7645, 5976, 4715, 3735, 2817, 2169, 1703, 1431, 1142, 825, 670, 570, 439, 350, 334, 254, 234])/100000.
    x = np.arange(150,1450,50)/1000.
    l_idx = find_nearest_idx(x,l)
    x_loc = index_to_xloc(l_idx) #x end of the line
    y_loc = y[l_idx] #y end of the line
    x_start = 1.0 #location of x start of the line
    y_start = 0.32 #location of y start of the line
    dist_from_line = 0.05

    dx = x_loc - x_start
    dy = y_loc - y_start
    d_mag = np.sqrt(dx * dx + dy * dy)
    new_x = x_loc - dx/ d_mag * dist_from_line #new location offset by distance from line
    new_y = y_loc - dy/ d_mag * dist_from_line

    # ax = pylab.axes()
    fig=Figure(figsize=(6.7,5.0),dpi=96)
    ax=fig.add_subplot(1,1,1)
    ax.plot(x, y, 'b', lw=1, label='damped sine', c='#0B6EC3')

    ax.set_title('Essay Word Count')
    ax.set_xlabel('# of words')
    ax.set_ylabel('# of users')
    
    ax.text(0.9, .35, "You are here")
    ax.plot([x_start, new_x], [y_start, new_y], '-k', lw=0.85)
    

    ax.set_xlim(0, 1.5)
    ax.set_ylim(0, 0.43)

    for i in [.150, .500,1.000]:
        ax.text(index_to_xloc(find_nearest_idx(x,i))-.03, -0.03, "%.0f" % (i*1000))

    #XKCDify the axes -- this operates in-place
    XKCDify(ax, xaxis_loc=0.0, yaxis_loc=0.0,
            xaxis_arrow='+-', yaxis_arrow='+-',
            expand_axes=True)
    # ax.add_patch(pylab.Circle((index_to_xloc(l_idx),y[l_idx]),radius=0.02, alpha=1.0, antialiased=True, fc="#BD362F", ec="none"))
    # pylab.show()
    canvas = FigureCanvas(fig)
    return fig
    
if __name__ == '__main__':
    plot_essay_len(800)
