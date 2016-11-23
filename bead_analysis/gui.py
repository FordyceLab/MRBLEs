# !/usr/bin/env python

# [Future imports]
# "print" function compatibility between Python 2.x and 3.x
from __future__ import print_function
# Use Python 3.x "/" for division in Pyhton 2.x
from __future__ import division

# [File header]     | Copy and edit for each file in this project!
# title             : gui.py
# description       : Bead Analysis GUI interface
# author            : Bjorn Harink
# credits           : Kurt Thorn, Huy Nguyen
# date              : 20161118
# version update    : 20161118
# version           : v0.1
# usage             : As part of Bead Analysis module
# notes             : Do not quick fix functions for specific needs, keep them general!
# python_version    : 2.7

# [TO-DO]

# [Modules]
# General Python
import sys
import os
import warnings
# GUI
import wx

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