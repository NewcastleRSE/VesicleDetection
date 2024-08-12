import matplotlib.pyplot as plt
import napari

# Create a function to images
def imshow(image):
    if len(image.shape) == 3:
        plt.figure(figsize=(10,10))
        for i in range(4):
            plt.subplot(1,4,i+1)
            plt.imshow(image[i])
        plt.show()

def imshow_napari(ret):
    viewer = napari.Viewer()
    viewer.add_image(data=ret['raw'].data, name='Raw')
    #viewer.add_image(data=zarr_data.target_data, name='Target', colormap='green', blending='additive')

    napari.run()