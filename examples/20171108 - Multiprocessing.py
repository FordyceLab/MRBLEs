# !/usr/bin/env python
#%%
import matplotlib.pyplot as plt
% matplotlib inline
import numpy as np
import xarray as xd
import bead_analysis as ba
ba.FindBeadsImagingP(18)

#%%
bead_objects = ba.FindBeadsImagingP(bead_size=18, border_clear=True)
# Channel(s) settings
ASSAY_CHANNELS = ['Cy5']  # Must be list!

# slice(Y1, Y2) and slice(X1, X2) Y and X are reversed in array since rows (Y) go first and columns go second (X)
# General Region or interest
CROPx = slice(312, 712)
CROPy = slice(312, 712)

# Setting bead image folder and image patterns. This will select all images following the pattern.
BEAD_IMAGE_FOLDER = r"C:\DATA\Huy\20170309 CN"
BEAD_IMAGE_PATTERNS = {"50 nM": r"20170309_CN_HQN106_PAP_50nM800ms_([1-9]|[1-9][0-9])_MMStack_Pos0.ome.tif",
                       "100 nM": r"20170309_CN_HQN106_PAP_100nM800ms_([1-9]|[1-9][0-9])_MMStack_Pos0.ome.tif",
                       "250 nM": r"20170309_CN_HQN106_PAP_250nM800msb_([1-9]|[1-9][0-9])_MMStack_Pos0.ome.tif",
                       "500 nM": r"20170309_CN_HQN106_PAP_500nM800msb_([1-9]|[1-9][0-9])_MMStack_Pos0.ome.tif",
                       "1000 nM": r"20170309_CN_HQN106_PAP_1uM800ms_([1-9]|[1-9][0-9])_MMStack_Pos0.ome.tif"}

# Search for files matching the patter in the bead image folder
bead_image_files = ba.ImageSetRead.scan_path(BEAD_IMAGE_FOLDER,
                                             BEAD_IMAGE_PATTERNS['50 nM'])
bead_image_obj = ba.ImageSetRead(bead_image_files)
bead_image_obj.crop_x = CROPx
bead_image_obj.crop_y = CROPy
print(bead_image_obj.c_names)  # Print channel names

#%%
fig_x = 10
plt.figure()
plt.imshow(bead_image_obj[fig_x, 'Brightfield'])

#%%
xdata = bead_objects.find(bead_image_obj[fig_x, 'Brightfield'])
plt.figure()
plt.imshow(xdata[0])

#%%
xdata = bead_objects.find(bead_image_obj[:, 'Brightfield'])

#%%
for x in range(xdata.sizes['f']):
    plt.figure(dpi=150)
    plt.imshow(xdata.loc[x, 'whole'].values)

#%%
plt.imshow(bead_objects.mask_inside[0])

