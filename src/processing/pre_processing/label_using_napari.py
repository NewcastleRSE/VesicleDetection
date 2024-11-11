import napari 
import glob
import numpy as np
import skimage 
import scipy.ndimage
import os 
from magicgui import magicgui
import pathlib

class GroundTruth():
    """"
        A class that will allow us to generate the ground truth vesicle spheres 
        around hand labelled vesicle centres. 

        Attributes
        -------------------
        pos_data (array): 
            An array labelling the vesicle centres for PC+ vesicles.
        neg_data (array): 
            An array labelling the vesicle centres for PC- vesicles.
        voxel_size (tuple(int)): 
            Tuple containing the (z,y,x) resolution.
        offset (tuple(int)): 
            The offset of the array. Required only for converting 
            result to zarr data. Default is (0,0,0).
        background_label (int):
            The background label value of the array. Required only for 
            converting result to zarr data. Default is 0. 
        num_classes (int): 
            The number of different label options, including background.
            Required only for converting result to zarr data. Default is
            3 (PC+, PC-, background)
        axes (list(str)): 
            List containing axes order for the data. Required only for converting 
            result to zarr data. Default is ['z','y','x'].
        balls (dict): 
            Dictionary who's key is the diameter of a ball and value is 
            the corresponding skimage ball (see 'draw_ball' function).
        min_distance (int): 
            The minimum distance between two vesicle centre labels to be 
            considered separate vesicle labels (i.e. independent spheres).
        kernel_shape (array):
            The size (in voxels) of the spheres to be drawn.
        ground_truth (array):
            The array containing the computed ground truth data.
    """
     
    def __init__(self,
                 pos_data, 
                 neg_data,
                 vesicle_diameter: int,
                 resolution: tuple, 
                 offset = (0,0,0), 
                 background_label = 0, 
                 num_classes = 3, 
                 axes =['z','y','x'], 
                 min_distance = 1):
        
        """
            Initialise a groundtruth object. 

            Parameters
            -------------------
            pos_data (array):
                Array containing hand labelled PC+ vesicle centres.
            neg_data (array):
                Array containing hand labelled PC- vesicle centres.
            vesicle_diameter (int):
                Requested real world diameter of vesicle spheres -- can be
                in any unit as long as matches 'resolution'. 
            resolution (tuple(int)):
                Resolution of the image -- can be in any unit as long as 
                matches 'vesicle_diameter'. Order should match 'axes'.
            offset:
                The offset of the array. Required only for converting 
                result to zarr data. Default is (0,0,0).
            background_label (int):
                The background label value of the array. Required only for 
                converting result to zarr data. Default is 0. 
            num_classes (int): 
                The number of different label options, including background.
                Required only for converting result to zarr data. Default is
                3 (PC+, PC-, background)
            axes (list(str)): 
                List containing axes order for the data. Required only for converting 
                result to zarr data. Default is ['z','y','x'].
            balls (dict): 
                Dictionary who's key is the diameter of a ball and value is 
                the corresponding skimage ball (see 'draw_ball' function).
            min_distance (int): 
                The minimum distance between two vesicle centre labels to be 
                considered separate vesicle labels (i.e. independent spheres).
        """

        self.pos_data = pos_data
        self.neg_data = neg_data
        self.voxel_size = resolution
        self.offset = offset
        self.background_label = background_label
        self.num_classes = num_classes
        self.axes = axes 
        self.balls = {}
        self.min_distance = min_distance

        diameter = np.array([vesicle_diameter, vesicle_diameter, vesicle_diameter])
        self.kernel_shape = np.ceil(diameter/self.voxel_size)
        
    def find_centres(self, data):
        """
            Find centres of hand labelled data. Convolves data with a ball 
            of size kernel_shape and looks for peak_local_max values. 

            Parameters
            -------------------
            data (array): 
                The hand labelled vesicle centre data.

            Returns 
            -------------------
            centre_indices (ndarry):
                Vesicle centre coordinates, as given by skimage.feature.peak_local_max.
        """

        kernel = skimage.morphology.ball(100).astype(np.float64) 
        kernel = skimage.transform.resize(kernel, self.kernel_shape).astype(np.float64)

        # Convolve the prediction with a spherical kernel
        data_convolved = scipy.ndimage.convolve(data*5, kernel)

        # Find the indices for the peak local maxima
        centre_indices = skimage.feature.peak_local_max(
                                                    data_convolved, 
                                                    min_distance = self.min_distance, 
                                                    labels= skimage.measure.label(data)
                                                    )

        

        return centre_indices
    
    def draw_ball(self, array, location, label=1):
        """
            Draw a ball of shape kernel_shape centred at location inside 
            array, with value label. The array itself will be edited.

            Parameters 
            -------------------
            array:
                The array in which the ball should be drawn. 
            location:
                The indices for the centre of the ball. 
            label (optional): 
                The label value the ball should take. Default=1.
        """

        diameter = self.kernel_shape
        diameter = tuple(int(x) for x in diameter)
        # Check to see if we already have a ball of this size
        if diameter not in self.balls:
            ball = skimage.morphology.ball(100, dtype=bool)
            ball = skimage.transform.resize(
                                            ball, 
                                            diameter, 
                                            order=0, 
                                            anti_aliasing=False).astype(bool) 
            self.balls[diameter] = ball
        
        else: 
            ball = self.balls[diameter]

        
        # Check for ball going past the boundary and trim ball accordingly. 
        slices_list = []
        ball_slices_list = []
        for loc, d, boundary, ball_shape in zip(location, diameter, array.shape, ball.shape):
            ball_start = (loc - d//2)
            ball_end = (loc - d//2 +d) 
            ball_trim_start = 0 
            ball_trim_end = ball_shape 

            if (ball_start < 0):
                ball_trim_start = -1*ball_start
                ball_start = 0
            if (ball_end > boundary):
                ball_trim_end = boundary - ball_end 
                ball_end = boundary
            
            slices_list.append(slice(ball_start, ball_end))
            ball_slices_list.append(slice(ball_trim_start, ball_trim_end))
        
        slices = tuple(slices_list)
        ball_slices = tuple(ball_slices_list)

        # Go to ball location in array and set values to label
        view = array[slices]
        sliced_ball = ball[ball_slices]
        view[sliced_ball] = label

    def compute_gt(self):
        """
            Computes the ground truth data, using the find_centres and 
            draw_ball functions. 

            Returns 
            -------------------
            self.ground_truth:
                An array containing the computed ground truth.
        """

        hand_labelled_data = self.pos_data + self.neg_data

        self.ground_truth = np.zeros(hand_labelled_data.shape, dtype=np.int64)

        vesicle_centres = self.find_centres(data = hand_labelled_data)
        for i in range (len(vesicle_centres)):
            loc = tuple(vesicle_centres[i])
            self.draw_ball(array=self.ground_truth, location=loc, label=hand_labelled_data[loc])

        return self.ground_truth

if __name__ == '__main__':

    # Load the napari viewer
    viewer = napari.Viewer()

    # Define our magicgui widget buttons
    
    @magicgui(call_button='Compute GT')
    def get_gt(
            vesicle_diameter = 300,
            resolution_z=60,
            resolution_y=60,
            resolution_x=60, 
            min_distance: int = 1) -> napari.types.LayerDataTuple:
        """
            Computes the ground truth labelling layer, from the pos and neg layers.
            The user sets the vesicle_diameter and the resolutions (in any consistent
            measurement unit) using the provided boxes. The minimum distance between 
            two labels to be considered independent vesicles is set using the min_distance
            input. 

            The ground truth label layer (called 'gt') is then generated by clicking the 
            "Compute GT" call button. 
        """
        
        pos = viewer.layers['pos']
        neg = viewer.layers['neg']
        ground_truth = GroundTruth(
                                pos_data = pos.data, 
                                neg_data = neg.data, 
                                vesicle_diameter = vesicle_diameter,
                                resolution = (resolution_z,resolution_y,resolution_x),
                                min_distance=min_distance)
        gt = ground_truth.compute_gt()

        # Return gt data as Napari label layer.
        return (gt, {'name': 'gt'}, 'labels')
    
    @magicgui(call_button='Load')
    def load_raw(raw_data_path = pathlib.Path('path/to/raw.tif')) -> napari.types.LayerDataTuple:
        """
            Widget to allow the user to load in the raw data that is to be labelled. The data
            must be a TIF file, and the path to this file provided. This can either be entered
            manually, or using the dictionary navigation button.

            Clicking the 'Load' call button will load the provided TIF file into a Napari image 
            layer with name 'raw'.
        """

        raw_data = np.array([skimage.io.imread(raw_data_path)]) 
        # Check for stacks of images
        if (raw_data.shape[0] == 1):
            raw_data = raw_data[0,:]

        if 'raw' in viewer.layers:
            return (raw_data, {'name': 'raw'}, 'image')
        
        else: 
            raw = viewer.add_image(data = raw_data, name='raw')
    
    @magicgui(call_button='Load')
    def load_labels(pos_data_path = pathlib.Path('path/to/pos.tif'), 
                    neg_data_path = pathlib.Path('path/to/neg.tif')):
        
        """
            A widget to allow the user to load saved pos and neg label layers. Both the 
            pos and neg files should be TIF files, and the paths can be provided either
            manually or using dictionary navigation buttons. 

            Clicking the 'Load' button will load the provided TIF files into Napari as
            the pos and/or neg label layers. If these already exist, they will be overwritten.
        """
        
        try:
            pos_data = skimage.io.imread(pos_data_path)
            # Check to see if pos label layer already exists
            if 'pos' in viewer.layers:
                pos = viewer.layers['pos']
                pos.data = pos_data 
            else:
                pos = viewer.add_labels(data=pos_data, name='pos')
        except:
            if str(pos_data_path) == 'path/to/pos.tif' or str(pos_data_path) == '':
                # If the path is unaltered or left blank, do nothing. Safetly to not accidently 
                # overwrite pos layer if only want to load neg layer.
                pass
            else:
                raise FileExistsError('Pos data location provided not suitable. Please try again.')
        try:
            neg_data = skimage.io.imread(neg_data_path)
            # Check to see if neg label layer already exists
            if 'neg' in viewer.layers:
                neg = viewer.layers['neg']
                neg.data = neg_data 
            else: 
                neg = viewer.add_labels(data=neg_data, name='neg')
        except:
            if str(neg_data_path) == 'path/to/neg.tif' or str(neg_data_path) == '':
                # If the path is unaltered or left blank, do nothing. Safetly to not accidently 
                # overwrite neg layer if only want to load pos layer.
                pass
            else:
                raise FileExistsError('Neg data location provided not suitable. Please try again.')
    
    @magicgui(save_location ={'mode': 'd'}, call_button='Save centres')
    def save_centres(save_location = pathlib.Path('save/location')):
        """
            Widget to allow user to save their pos and neg label layers as TIF files. The 
            dictionary in which they wish to save their files is provided either manually, 
            or using the navigation button. 

            Clicking the 'Save centres' button will generate two files, pos.tif and neg.tif, 
            within the provided directory. Existing files with those names will be overwritten.
        """

        pos = viewer.layers['pos']
        neg = viewer.layers['neg']

        if not os.path.exists(save_location):
            os.makedirs(save_location)

        pos_data = pos.data
        neg_data = neg.data.astype(np.uint16)

        skimage.io.imsave(f'{save_location}/pos.tif', pos_data)
        skimage.io.imsave(f'{save_location}/neg.tif', neg_data)
            
    @magicgui(call_button='Clear centres')
    def clear_centres(Pos: bool = False, 
                      Neg: bool = False):
        """
            A widget to allow the user to generate a clean pos and/or neg label layer. Only
            layers that have the corresponding tick box checked will be affected. This widget 
            can be used to generate the pos and neg label layers at the start, if previously 
            saved ones are not to be loaded. 
        """
        
        raw_layer = viewer.layers['raw']
        if Pos == True:
            if 'pos' in viewer.layers:
                pos = viewer.layers['pos']
                pos.data = np.zeros(raw_layer.data.shape, dtype=np.int64)
            else:
                pos = viewer.add_labels(data = np.zeros(raw_layer.data.shape, dtype=np.int64), name='pos')

        if Neg == True:
            if 'neg' in viewer.layers:
                neg = viewer.layers['neg']
                neg.data = np.zeros(raw_layer.data.shape, dtype=np.int64)
            else: 
                neg = viewer.add_labels(data = np.zeros(raw_layer.data.shape, dtype=np.int64), name='neg')

    # Add widgets to napari window
    viewer.window.add_dock_widget(get_gt)
    viewer.window.add_dock_widget(load_raw)
    viewer.window.add_dock_widget(load_labels)
    viewer.window.add_dock_widget(save_centres)
    viewer.window.add_dock_widget(clear_centres)

    napari.run()