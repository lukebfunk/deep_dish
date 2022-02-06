import argparse
import napari
import numpy as np
import skimage
import zarr

parser = argparse.ArgumentParser()
parser.add_argument('store',type=str)
parser.add_argument('--method',type=str)
args = parser.parse_args()

viewer = napari.Viewer()

print("Reading cells...")
g = zarr.open(args.store)[args.method]

# dims: image index, channels, y, x
rows = 2
cols = 2
shape_ = g['real'].shape
montage = np.zeros(shape_[:-3]+(shape_[-2]*rows,shape_[-1]*cols),dtype=g['real'].dtype)
montage[...,:shape[-2],:shape[-1]] = g['real'][:]
montage[...,:shape[-2],shape[-1]:] = g['fake'][:]
montage[...,shape[-2]:,:shape[-1]] = g['mask_weight'][:]
montage[...,shape[-2]:,shape[-1]:] = g['hybrid'][:]
# dims: image index, channels, montage y, montage x
viewer.add_image(montage,channel_axis=1)


# print(images.shape)
# print(type(cells))

print("Opening napari...")
# viewer = napari.view_image(images,channel_axis=1)
# viewer.add_labels(cells)
napari.run()
