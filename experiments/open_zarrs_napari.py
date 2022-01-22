import argparse
import napari
import numpy as np
import zarr

parser = argparse.ArgumentParser()
parser.add_argument('stores',nargs='+',type=str)
parser.add_argument('--crop',default=None,type=int)
parser.add_argument('--channel_axis',default=None,type=int)
args = parser.parse_args()

viewer = napari.Viewer()

crop = args.crop

for store in args.stores:
    print("Reading cells...")
    z= zarr.open(store)
    for n,im in z.arrays():
        print(im.shape)
        if (crop is not None)&(min(im.shape[-2:])>crop):
            crop = (np.array(im.shape[-2:])-np.array([crop,crop]))//2
            im = im[...,crop[0]:-crop[0],crop[1]:-crop[1]]
        viewer.add_image(im,channel_axis=args.channel_axis,name=n)


# print(images.shape)
# print(type(cells))

print("Opening napari...")
# viewer = napari.view_image(images,channel_axis=1)
# viewer.add_labels(cells)
napari.run()
