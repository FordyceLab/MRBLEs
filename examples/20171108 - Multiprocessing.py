# !/usr/bin/env python
#%%
import numpy as np
import xarray as xd
import mrbles as ba
import matplotlib
import matplotlib.pyplot as plt
% matplotlib tk

#%%
bead_objects = ba.FindBeadsImaging(bead_size=18, border_clear=True, circle_size=360)
# Channel(s) settings
ASSAY_CHANNELS = ['Cy5_5%']  # Must be list!

# slice(Y1, Y2) and slice(X1, X2) Y and X are reversed in array since rows (Y) go first and columns go second (X)
# General Region or interest
CROPx = slice(10, 990)
CROPy = slice(10, 990)

# Setting bead image folder and image patterns. This will select all images following the pattern.
BEAD_IMAGE_FOLDER = r"data"
BEAD_IMAGE_PATTERN =

# Search for files matching the patter in the bead image folder
bead_image_files = ba.ImageSetRead.scan_path(BEAD_IMAGE_FOLDER,
                                             BEAD_IMAGE_PATTERN)
bead_image_obj = ba.ImageSetRead(bead_image_files, output='xr')
bead_image_obj.crop_x = CROPx
bead_image_obj.crop_y = CROPy
print(bead_image_obj.c_names)  # Print channel names
bead_image_obj._dataframe.rename()

#%%
fig_x = 1
test_find = ba.pipeline.Find(bead_size=18, border_clear=True, circle_size=360)
x_comb = test_find.find(bead_image_obj[:, 'Brightfield'], bead_image_obj.data)


#%%
folders = {
    'files1': 'data',
    'files2': 'data'
}
files = {
    'files1': r"peptide_biotin_streptavidin_([0-9][0-9])_MMStack_Pos0.ome.tif",
    'files2': r"peptide_biotin_streptavidin_([0-9][0-9])_MMStack_Pos0.ome.tif"
}
filez = {key1: ba.ImageSetRead.scan_path(folders[key1], file) for key1, file in files.items()}

#%%
print(filez['files2'])

#%%
fig_x = 0
plt.figure()
plt.imshow(bead_image_obj[fig_x, 'Brightfield'])

#%%
bead_image_obj.crop_x

#%%
bead_image_obj.data

#%%
bead_objects.find(bead_image_obj[fig_x, 'Brightfield'])
xdata = bead_objects.xdata
plt.figure()
plt.imshow(xdata[0])

#%%
bead_objects.crop_x = CROPx
bead_objects.crop_y = CROPy
bead_objects.find(bead_image_obj[:, 'Brightfield'])
xdata = bead_objects.xdata

#%%
for x in range(xdata.sizes['f']):
    plt.figure(dpi=113)
    plt.imshow(xdata.loc[x, 'image_roi'].values)

#%%
bead_objects.xdata

#%%
plt.figure(dpi=113)
bead_objects.show_image_overlay(bead_image_obj[0, 'Brightfield'],
                                bead_objects.mask('ring')[0],
                                alpha=0.4)

#%%
#np.unique(bead_objects.mask_bkg[0])
#bead_objects.bead_num
bead_objects.bead_dims.to_csv(r"D:\test.csv")
#bead_objects.bead_labels

#%%
test_con = bead_image_obj.data.combine_first(bead_objects.data).astype(float)

#%%
plt.figure()
plt.imshow(test_con.loc[0, 'ring'])

#%%
bead_objects.data.loc[0, 'ring']

#%%
plt.figure(dpi=113)
bead_objects.show_image_overlay(test_con.loc[0, 'image_roi'],
                                test_con.loc[0, 'ring'],
                                alpha=0.4)

#%%
test = ba.pipeline.Images(folders, files)
test.load()

#%%
print(test._dataframe)
xdict = test._dataframe

#%%
test.crop_x = slice(90,990)
test.crop_y = slice(90,990)
test.ndata.shape
#%%
xd.concat()