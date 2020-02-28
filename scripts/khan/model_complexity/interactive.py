import numpy as np
import matplotlib.pyplot as plt
from matplotlib.text import Annotation
import seaborn as sns
sns.set_style('whitegrid')
    
class SnaptoCursor(object):

    def __init__(self, ax, x, y1, y2, label):
        self.ax = ax
        self.lx = ax.axhline(color='k')  # the horiz line
        self.ly = ax.axvline(color='k')  # the vert line
        self.x = x
        self.y1 = y1
        self.y2 = y2
        self.label = label

        # text location in axes coords
        self.txt1 = ax.text(0.1, 0.9, '', transform=ax.transAxes)

    
    def mouse_move(self, event):
        if not event.inaxes:
            return

        x, y1, y2 = event.xdata, event.ydata, event.ydata
        indx = min(np.searchsorted(self.x, x), len(self.x) - 1)
        x = self.x[indx]
        y1 = self.y1[indx]
        y2 = self.y2[indx]
        label_val = self.label[indx]

        # update the line positions
        self.ly.set_xdata(x)

        self.txt1.set_text('Set=%d :%s' % (x, label_val))
        self.ax.figure.canvas.draw()


class Interactive_Plot(SnaptoCursor):

    def __init__(self, x, y1, y2, model1, model2, label):
        
        self.x = x
        self.y1 = y1
        self.y2 = y2
        self.label = label
        self.model1 = model1
        self.model2 = model2


    def fig_plot(self):
       
        x = self.x
        y1 = self.y1
        y2 = self.y2
        annot_dict = self.label
        model1 = self.model1
        model2 = self.model2

        fig = plt.figure(figsize=(15,8))
        ax = fig.add_subplot(111)
        ax.scatter(x, y1, color = 'red', label = model1)
        ax.scatter(x, y2, color = 'blue', label = model2)
        ax.legend()
        plt.xlabel('Parameter set', fontsize=30)
        plt.ylabel(r'EO ($\times 10^4$)', fontsize=30)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        snap_cursor = SnaptoCursor(ax, x, y1, y2, annot_dict)
        fig.canvas.mpl_connect('motion_notify_event', snap_cursor.mouse_move)
        plt.show()

