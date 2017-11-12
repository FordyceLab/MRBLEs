# !/usr/bin/env python

"""
Gui Classes and Functions
=========================

This file stores the GUI classes and functions for the MRBLEs Analysis module.
"""

# [Future imports]
from __future__ import print_function, division

# [File header]     | Copy and edit for each file in this project!
# title             : gui.py
# description       : MRBLEs GUI tools
# author            : Bjorn Harink
# credits           :
# date              : 20161118

# [Modules]
# General Python
import sys
import os
# GUI
import wx

# Function compatibility between Python 2.x and 3.x
if sys.version_info < (3, 0):
    from future.standard_library import install_aliases
    from __builtin__ import *  # NOQA
    install_aliases()


### Classes


class MainWindow(wx.Frame):
    """This is the main window for the Flow Module of the Microfluidic Control program.
    """
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.SetTitle('Bead Analysis')
        self.init()

    def init(self):
        # MainSizer and Tabs
        self.mainSizer = wx.BoxSizer(wx.HORIZONTAL)
        tabs = wx.Notebook(self)

        self.bead_find = BeadFinder(self)
        tabs.AddPage(self, self.bead_find, u"Bead Finder", False)

    def __del__(self):
        self.Close()

    def close(self, event):
        self.Show(False)

    def show(self, event):
        self.Show(True)

class BeadFinder(wx.Panel):
    def __init__(self, parent=None):
        super(BeadFinder, self).__init__(parent)

        # mainSizer
        mainSizer = wx.BoxSizer(wx.HORIZONTAL)

        imageSizer = wx.BoxSizer(wx.VERTICAL)
        self.bead_image = wx.EmptyBitmap(900, 900)
        self.beadImageFrame = wx.StaticBitmap(self, bitmap=self.bead_image)
        imageSizer.Add(self.beadImageFrame, 1, wx.EXPAND|wx.ALL|wx.SHAPED, 0)
        imageSizer.Fit(self)
        mainSizer.Add(imageSizer, 1, wx.EXPAND|wx.ALL|wx.SHAPED, 0)

    def update(self, event):
        self.image_update()
        self.Fit()
        self.Parent.Parent.update(event)

    def image_update(self):
        image = self.resize_image(self.bead_image, min(self.Size))
        self.beadImage.SetBitmap(image)

    @staticmethod
    def resize_image(image_path, max_size):
        image = wx.Image(image_path, wx.BITMAP_TYPE_ANY)
        max_size = max_size
        W = image.GetWidth()
        H = image.GetHeight()
        if W > H:
            new_W = max_size
            new_H = max_size * H / W
        else:
            new_H = max_size
            new_W = max_size * W / H
        image = wx.BitmapFromImage(image.Rescale(new_W,new_H, wx.IMAGE_QUALITY_HIGH))
        return image

def main():
    wxapp = wx.App()
    locale = wx.Locale(wx.LANGUAGE_ENGLISH)

    main_gui = MainWindow(None)
    main_gui.show()
    wxapp.MainLoop()

    return 0

# Main loop
if __name__ == '__main__':
    status = main()
    sys.exit(status)



#################################### TO-DO
from __future__ import print_function

import sys
import time
import os
import gc
import matplotlib
matplotlib.use('WXAgg')
import matplotlib.cm as cm
import matplotlib.cbook as cbook
from matplotlib.backends.backend_wxagg import Toolbar, FigureCanvasWxAgg
from matplotlib.figure import Figure
import numpy as np

import wx
import wx.xrc as xrc

ERR_TOL = 1e-5  # floating point slop for peak-detection


matplotlib.rc('image', origin='lower')


class PlotPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1)

        self.fig = Figure((5, 4), 75)
        self.canvas = FigureCanvasWxAgg(self, -1, self.fig)
        self.toolbar = Toolbar(self.canvas)  # matplotlib toolbar
        self.toolbar.Realize()
        # self.toolbar.set_active([0,1])

        # Now put all into a sizer
        sizer = wx.BoxSizer(wx.VERTICAL)
        # This way of adding to sizer allows resizing
        sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        # Best to allow the toolbar to resize!
        sizer.Add(self.toolbar, 0, wx.GROW)
        self.SetSizer(sizer)
        self.Fit()

    def init_plot_data(self):
        a = self.fig.add_subplot(111)

        x = np.arange(120.0) * 2 * np.pi / 60.0
        y = np.arange(100.0) * 2 * np.pi / 50.0
        self.x, self.y = np.meshgrid(x, y)
        z = np.sin(self.x) + np.cos(self.y)
        self.im = a.imshow(z, cmap=cm.RdBu)  # , interpolation='nearest')

        zmax = np.amax(z) - ERR_TOL
        ymax_i, xmax_i = np.nonzero(z >= zmax)
        if self.im.origin == 'upper':
            ymax_i = z.shape[0] - ymax_i
        self.lines = a.plot(xmax_i, ymax_i, 'ko')

        self.toolbar.update()  # Not sure why this is needed - ADS

    def GetToolBar(self):
        # You will need to override GetToolBar if you are using an
        # unmanaged toolbar in your frame
        return self.toolbar

    def OnWhiz(self, evt):
        self.x += np.pi / 15
        self.y += np.pi / 20
        z = np.sin(self.x) + np.cos(self.y)
        self.im.set_array(z)

        zmax = np.amax(z) - ERR_TOL
        ymax_i, xmax_i = np.nonzero(z >= zmax)
        if self.im.origin == 'upper':
            ymax_i = z.shape[0] - ymax_i
        self.lines[0].set_data(xmax_i, ymax_i)

        self.canvas.draw()

    def onEraseBackground(self, evt):
        # this is supposed to prevent redraw flicker on some X servers...
        pass


class MyApp(wx.App):
    def OnInit(self):
        xrcfile = cbook.get_sample_data('embedding_in_wx3.xrc',
                                        asfileobj=False)
        print('loading', xrcfile)

        self.res = xrc.XmlResource(xrcfile)

        # main frame and panel ---------

        self.frame = self.res.LoadFrame(None, "MainFrame")
        self.panel = xrc.XRCCTRL(self.frame, "MainPanel")

        # matplotlib panel -------------

        # container for matplotlib panel (I like to make a container
        # panel for our panel so I know where it'll go when in XRCed.)
        plot_container = xrc.XRCCTRL(self.frame, "plot_container_panel")
        sizer = wx.BoxSizer(wx.VERTICAL)

        # matplotlib panel itself
        self.plotpanel = PlotPanel(plot_container)
        self.plotpanel.init_plot_data()

        # wx boilerplate
        sizer.Add(self.plotpanel, 1, wx.EXPAND)
        plot_container.SetSizer(sizer)

        # whiz button ------------------
        whiz_button = xrc.XRCCTRL(self.frame, "whiz_button")
        whiz_button.Bind(wx.EVT_BUTTON, self.plotpanel.OnWhiz)

        # bang button ------------------
        bang_button = xrc.XRCCTRL(self.frame, "bang_button")
        bang_button.Bind(wx.EVT_BUTTON, self.OnBang)

        # final setup ------------------
        sizer = self.panel.GetSizer()
        self.frame.Show(1)

        self.SetTopWindow(self.frame)

        return True

    def OnBang(self, event):
        bang_count = xrc.XRCCTRL(self.frame, "bang_count")
        bangs = bang_count.GetValue()
        bangs = int(bangs) + 1
        bang_count.SetValue(str(bangs))

if __name__ == '__main__':
    app = MyApp(0)
    app.MainLoop()