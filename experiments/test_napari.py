import napari
import zarr

print("Reading cells...")
cells = zarr.open('/nrs/funke/funkl/data/cell_patches/CCT7.zarr')['CTTGTCCAAACTCCCCATTG/mitotic/images'][:]

print(cells.shape)
print(type(cells))

print("Opening napari...")
viewer = napari.view_image(cells)
napari.run()
