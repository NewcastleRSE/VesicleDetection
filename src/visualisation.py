import matplotlib.pyplot as plt 

# Create a function to images
def imshow(image):
    if len(image.shape) == 3:
        plt.figure(figsize=(10,10))
        for i in range(4):
            plt.subplot(1,4,i+1)
            plt.imshow(image[i])
        plt.show()