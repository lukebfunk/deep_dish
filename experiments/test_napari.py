import napari
import zarr

print("Reading cells...")
images = zarr.open('/nrs/funke/funkl/data/cell_patches/DTL.zarr')['ACAGCTGTATTCTGTACGGG/interphase/images'][:]
cells = zarr.open('/nrs/funke/funkl/data/cell_patches/DTL.zarr')['ACAGCTGTATTCTGTACGGG/interphase/cells'][:]

print(images.shape)
print(type(cells))

print("Opening napari...")
viewer = napari.view_image(images,channel_axis=1)
viewer.add_labels(cells)
napari.run()
