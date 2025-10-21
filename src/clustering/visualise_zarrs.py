import zarr
import napari
import sys
import h5py

if len(sys.argv) < 1:
    print("Usage: python visualise_zarrs.py <zarr_path> <zarr_path1> <zarr_path2> ...")
    sys.exit(1)
else:
    paths = sys.argv[1:]

viewer = napari.Viewer()
options = ["raw", "masked_raw", "mask", "masked", "masks"]

for path in paths:
    print(f"Visualizing {path}")
    if path.endswith('.h5'):
        print("Loading h5 file")
        f = h5py.File(path, 'r')
        # Assume dataset name is same as file name without extension
        dset_name = path.split('/')[-1].replace('.h5', '')
        if dset_name in f:
            zarrdat = f[dset_name][:]
            option = ''
        else:
            # If not found, just take the first dataset
            first_key = list(f.keys())[0]
            zarrdat = f[first_key][:]
            option = ''
        f.close()
    else:
        try:
            image = zarr.open(path, mode='r')        
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
        except Exception as e:
            print(f"Error loading {path} as zarr: {e}")
            pass

    viewer.add_image(data=zarrdat, name=path.split("/")[-1]+option, blending='additive', colormap='blue')
 
napari.run()


