import zarr
import napari
import sys

if len(sys.argv) < 1:
    print("Usage: python visualise_zarrs.py <zarr_path> <zarr_path1> <zarr_path2> ...")
    sys.exit(1)
else:
    zarrpaths = sys.argv[1:]

viewer = napari.Viewer()
options = ["raw", "masked_raw", "mask", "masked", "masks"]

for zarrpath in zarrpaths:
    print(f"Visualizing {zarrpath}")
    image = zarr.open(zarrpath, mode='r')
    if 'raw' in image:
        zarrdat = image['raw'][:,:,:]
        option = '_raw'
    elif 'masked_raw' in image:
        zarrdat = image['masked_raw'][:,:,:]
        option = '_masked'
    elif 'mask' in image:
        zarrdat = image['mask'][:,:,:]
        option = '_mask'
    elif 'masked' in image:
        zarrdat = image['masked'][:,:,:]
        option = '_masked'
    elif 'masks' in image:
        zarrdat = image['masks'][:,:,:]
        option = '_masks'
    else:
        zarrdat = image[:,:,:]
        option = ''
       
    viewer.add_image(data=zarrdat, name=zarrpath.split("/")[-1]+option, blending='additive', colormap='blue')
 
napari.run()


