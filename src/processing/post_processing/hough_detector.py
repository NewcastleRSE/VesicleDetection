import numpy as np
import scipy.ndimage 
import skimage.feature 
import skimage.morphology
import zarr
import gunpowder as gp

from config.load_configs import POST_PROCESSING_CONFIG

class HoughCandidate:
        def __init__(self, location, maxima, label):
            self.location = location
            self.score = maxima 
            self.label = label

class HoughDetector:

    def __init__(self, pred_pos, pred_neg, voxel_size, combine_pos_neg = POST_PROCESSING_CONFIG.combine_pos_neg):

        # self.pred_pos = pred_pos
        # self.pred_neg = pred_neg
        self.pred_pos_data = pred_pos
        self.pred_neg_data = pred_neg
        self.combine_pos_neg = combine_pos_neg
        self.voxel_size = voxel_size
        self.balls = {}

    def hough_prediction(self, threshold):

        self.roi = self.pred_pos_data.shape

        # Define our spherical kernel for convolution
        diameter = np.array([300.0, 300.0, 300.0])
        self.kernel_shape = np.ceil(diameter/self.voxel_size)
        kernel = skimage.morphology.ball(100).astype(np.float64) 
        kernel = skimage.transform.resize(kernel, self.kernel_shape).astype(np.float64)

        if self.combine_pos_neg: 
            # Get maxima and index locations 
            maxima, maxima_indices = self.hough_transformation(
                                                            self.pred_pos_data + self.pred_neg_data,
                                                            kernel,
                                                            threshold
                                                            )
            # Get maxima for positive and negative
            maxima_pos = self.pred_pos_data[
                                            maxima_indices[:,0],
                                            maxima_indices[:,1],
                                            maxima_indices[:,2]
                                            ]
            maxima_neg = self.pred_neg_data[
                                            maxima_indices[:,0],
                                            maxima_indices[:,1],
                                            maxima_indices[:,2]
                                            ]
            
            # Get labels
            labels = (maxima_pos < maxima_neg).astype(np.uint8) + 1

            # Get candidates and sort in decreasing order
            candidates = {}
            for i in range(len(maxima)):
                candidates.update( {(labels[i], tuple(maxima_indices[i])) : maxima[i]} )

            self.candidates = sorted(candidates.items(), key = lambda x: x[1], reverse=True)
        
        else: 
            maxima_pos, maxima_indices_pos = self.hough_transformation(
                                                            self.pred_pos_data, 
                                                            kernel, 
                                                            threshold
                                                            )
            
            maxima_neg, maxima_indices_neg = self.hough_transformation(
                                                            self.pred_neg_data, 
                                                            kernel, 
                                                            threshold
                                                            )

            candidates = {}
            # Get positive candidates and assign label 1
            for i in range(len(maxima_pos)):
                candidates.update( {(1, tuple(maxima_indices_pos[i])) : maxima_pos[i]} )

            # Get negative candidates and assign label 2
            for i in range(len(maxima_neg)):
                candidates.update( {(2, tuple(maxima_indices_neg[i])) : maxima_neg[i]} )
            
            # Sort all candidates in decreasing order
            self.candidates = sorted(candidates.items(), key = lambda x: x[1], reverse=True)

    def hough_transformation(self, prediction, kernel, threshold):

        # Convolve the prediction with a spherical kernel
        prediction_convolved = scipy.ndimage.convolve(prediction, kernel)
        # Find the indices for the peak local maxima
        maxima_indices = skimage.feature.peak_local_max(
                                                    prediction_convolved, 
                                                    min_distance = 2, 
                                                    threshold_abs = threshold
                                                    )
        # Get the maxima values at peak local maxima indices 
        maxima = prediction_convolved[maxima_indices[:,0], maxima_indices[:,1], maxima_indices[:,2]]

        return maxima, maxima_indices
    
    def draw_ball(self, array, location, diameter, label=1):
        
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

        # Find the slice for the ball
        slices = tuple( 
                    slice(loc - d//2, loc -d//2 +d)
                    for loc, d in zip(location, diameter)
                    )

        # Go to ball location in array and set values to label
        view = array[slices]
        view[ball] = label

    def probe_candidate(self, reject_map, location):

        # Check for existence of other predictions at this location
        if reject_map[location].item():
            return False

        diameter = 2*self.kernel_shape

        # Check to see if detection would leave the bounds of the image.
        bounds_check = [
            loc < d/2
            for loc, d in zip(location, diameter)
        ]
        bounds_check += [
            loc >= s - d/2
            for loc, d, s in zip(
                location,
                diameter,
                reject_map.shape)
        ]
        if any(bounds_check):
            return False

        self.draw_ball(reject_map, location, diameter)

        return True
    
    def prediction(self, accepted_candidates):

        # Start with all background
        self.prediction_result = np.zeros(self.pred_pos_data.shape, dtype=np.int64)
        
        # Add labels as per accepted candidates. 
        for candidate in accepted_candidates:
            self.draw_ball(
                            array=self.prediction_result, 
                            location=candidate.location, 
                            diameter=self.kernel_shape, 
                            label=candidate.label
                            )
    
    def process(self, maxima_threshold = POST_PROCESSING_CONFIG.maxima_threshold):

        # Define reject map. This will be updated with "True" values 
        # In locations where vesicles have been detected.
        reject_map = np.zeros(self.pred_pos_data.shape, dtype=bool)

        self.hough_prediction(maxima_threshold)

        self.accepted_candidates = []
        for candidate in self.candidates:
            if candidate[1] < maxima_threshold:
                break 
            
            # Chek to see if candidate is valid
            accepted = self.probe_candidate(
                                            reject_map=reject_map,
                                            location=candidate[0][1]
                                            )
            
            # Generate instance of HoughCandidate and save accepted candidates.
            if accepted: 
                self.accepted_candidates.append(
                                            HoughCandidate(
                                                        location = candidate[0][1], 
                                                        maxima = candidate[1], 
                                                        label = candidate[0][0]
                                                        )
                                            )
        # Generate the prediction
        self.prediction(self.accepted_candidates)
            

if __name__ == "__main__":

    prediction_data_path = input("Provide path to prediction: ")

    f = zarr.open(prediction_data_path, mode = 'r+')

    pred_pos = f['Positive']
    pred_neg = f['Negative']
    
    hough_detection = HoughDetector(pred_pos=pred_pos, pred_neg = pred_neg, combine_pos_neg=True)

    hough_detection.process()

    print(len(hough_detection.accepted_candidates))
    print(len(hough_detection.candidates))

    f['Hough_transformed'] = hough_detection.prediction_result

    for atr in f['Positive'].attrs:
        f['Hough_transformed'].attrs[atr] = f['Positive'].attrs[atr]
