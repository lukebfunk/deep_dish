import argparse
import napari
import zarr


print("Reading cells...")
images = zarr.open('/nrs/funke/funkl/data/example_val.zarr')['images'][0:100]

print(images.shape)
# print(type(cells))

print("Opening napari...")
viewer = napari.view_image(images,channel_axis=1)
# viewer.add_labels(cells)
napari.run()
