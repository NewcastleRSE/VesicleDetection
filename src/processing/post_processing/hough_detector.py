import numpy as np
import scipy.ndimage 
import skimage.feature 
import skimage.morphology
import zarr

from config.load_configs import POST_PROCESSING_CONFIG

class HoughCandidate:
        def __init__(self, location, maxima, label):
            """
                Vesicle candidates after post processing.

                Attributes 
                -------------------
                location:
                    The location of the centre of the vesicle candidate.
                score: 
                    The confidence score for this candidate to be a vesicle.
                label: 
                    The label of the vesicle.
            """
            self.location = location
            self.score = maxima 
            self.label = label

class HoughDetector:

    def __init__(self, 
                 pred_pos, 
                 pred_neg, 
                 voxel_size, 
                 combine_pos_neg = POST_PROCESSING_CONFIG.combine_pos_neg,
                 bias = 1):
        """
            Post processing class for output of vesicle detection model. 

            Attributes 
            -------------------
            pred_pos_data:
                The array containing the predicted probability distribution for PC+ vesicles in the image.
            pred_neg_data:
                The array containing the predicted probability distribution for PC- vesicles in the image.
            combine_pos_neg:
                Whether to combine the probabilities for PC+ and PC- detection. If set to true, this will 
                result in a post processing procedure that favours finding vesicle existance and later labelling, 
                rather than looking for PC+ and PC- vesicles independently. 
            bias:
                A bias factor for the labelling of vesicle candidates. Enters as maxima_pos < bias * maxima_neg, 
                so bias greater than 1 favours PC- labelling while bias less than 1 favours PC+. Only used when 
                combine_pos_neg = True.
            voxel_size:
                The voxel size of the image.
            balls:
                A dictionary who's key is the diameter of a ball and who's value is corresponding skimage ball.
            roi:
                The region of interest, i.e. the shape of the prediction. 
            kernel_shape:
                The shape of the balls translated into voxel sizes.
            candidates:
                All vesicle candidates. 
            accepted_candidates:
                Vesicle candidates that are accepted after elimination process (e.g. thresholding and asserting
                non-overlap of vesicles).
            prediction_result:
                The array containing the vesicle prediction after post processing.     
        """

        self.pred_pos_data = pred_pos
        self.pred_neg_data = pred_neg
        self.combine_pos_neg = combine_pos_neg
        self.voxel_size = voxel_size
        self.balls = {}
        self.bias = bias

    def hough_prediction(self, threshold):
        """
            Obtain the candidate vesicles after thresholding. 

            Parameters
            -------------------
            threshold (float):
                The threshold the peak_local_max must exceed to be considered a candidate.
        """

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
            labels = (maxima_pos < self.bias * maxima_neg).astype(np.uint8) + 1

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
        """
            Hough transformation of the predicition: prediction is convolved with the kernel
            and then peak_local_max coordinates and values obtained. 

            Parameters
            -------------------
            prediction (array):
                The prediction that is to be hough transformed. 
            kernel (array):
                The kernel to convolve the data with. For spherical vesicle searches, kernel
                should be a sphere with diameter given by expected vesicle size. 
            threshold (float):
                The minimum intensity a peak must have in the convolved image to be considered 
                a suitable vesicle candidate.

            Returns 
            -------------------
            maxima (list):
                The peak values at the vesicle candidate centres in the convolved image. 
            maxima_indices (list):
                The indices of the maxima peaks. 
        """

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
        """
            Method for drawing a ball inside an array of set diameter and label. 

            Parameters
            -------------------
            array:
                The array in which to draw the ball. 
            location: 
                The indices for the centre of the ball. 
            label:
                The value of the ball within the array. 
        """
        
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

    def probe_candidate(self, reject_map, location):
        """
            Check whether a vesicle candidate should be rejected.

            Parameters 
            -------------------
            reject_map:
                This array will store the locations of already accepted vesicles. 
                Allows to check for non-overlapping of vesicles. 
            location:
                The index location for the centre of the candidate vesicle. 
        """

        # Check for existence of other predictions at this location
        if reject_map[location].item():
            return False

        diameter = 2*self.kernel_shape

        self.draw_ball(reject_map, location, diameter)

        return True
    
    def prediction(self, accepted_candidates):
        """
            Draws balls at the locations of accepted vesicle candidates. 

            Parameters
            -------------------
            accepted_candidates (list(HoughCandidates)):
                List of accepted vesicle candidates. Each candidate should be an instance 
                of the HoughCandidate class, to allow for extracting location and label.
        """

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

        """
            Run the post processing process. 

            Parameters
            -------------------
            maxima_threshold (float):
                The threshold used to determine vesicle candidates. Default is 
                set using the post_processing_config.yaml file.
        """

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
