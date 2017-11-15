# !/usr/bin/env python
#%%
import matplotlib
import matplotlib.pyplot as plt
% matplotlib inline
import numpy as np
import xarray as xd
import mrbles as ba

#%%
bead_objects = ba.FindBeadsImagingP(bead_size=18, border_clear=True, circle_size=340)
# Channel(s) settings
ASSAY_CHANNELS = ['Cy5_5%']  # Must be list!

# slice(Y1, Y2) and slice(X1, X2) Y and X are reversed in array since rows (Y) go first and columns go second (X)
# General Region or interest
CROPx = slice(10, 990)
CROPy = slice(10, 990)

# Setting bead image folder and image patterns. This will select all images following the pattern.
BEAD_IMAGE_FOLDER = r"data"
BEAD_IMAGE_PATTERN = r"peptide_biotin_streptavidin_([0-9][0-9])_MMStack_Pos0.ome.tif"

# Search for files matching the patter in the bead image folder
bead_image_files = ba.ImageSetRead.scan_path(BEAD_IMAGE_FOLDER,
                                             BEAD_IMAGE_PATTERN)
bead_image_obj = ba.ImageSetRead(bead_image_files)
bead_image_obj.crop_x = CROPx
bead_image_obj.crop_y = CROPy
print(bead_image_obj.c_names)  # Print channel names

#%%
fig_x = 0
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
    plt.imshow(xdata.loc[x, 'bkg'].values)

#%%
plt.figure(dpi=150)
plt.imshow(bead_objects.mask_bkg[0])

#%%
#np.unique(bead_objects.mask_bkg[0])
bead_objects.bead_num
#bead_objects.bead_labels
