
import os
from datetime import datetime
import numpy as np
import skimage.io
import zarr
import torch

from src.data_loader import EMData
from src.processing.predict import Prediction
from src.model.model import DetectionModel
from src.processing.post_processing.hough_detector import HoughDetector
from src.directory_organisor import create_unique_directory_file
from src.visualisation import imshow_napari_prediction

from config.load_configs import TRAINING_CONFIG


class Apply:
    def __init__(self, model_checkpoint, label_background=0):
        """

        :param model_checkpoint: Path to the model that should be used for prediction.
        :type model_checkpoint: str
        :param label_background: The label value for the background pixels/voxels. Default is 0.
        :type label_background: int | None
        """

        self.model_checkpoint = model_checkpoint

        if label_background is None:
            self.label_background = 0
        elif isinstance(label_background, int):
            self.label_background = label_background
        else:
            raise TypeError("label_background must be an int or None")

    def __call__(self, *args, **kwargs):

        return self.single_image_multi_biases(*args, **kwargs)

    def single_image_single_bias(
            self, zarr_path, bias=1.0,
            save_all_labels_in_one_tiff_file=False, tiff_file_name_of_all_labels=None,
            save_different_labels_in_different_tiff_files=False,
            tiff_file_names_of_different_labels= None,
            dtype_labels=None, show=False):

        """Use a pretrained vesicle detection model to predict vesicles in unlabelled data by using a single bias.

        :param zarr_path: Path to the zarr group that contains the 'predict' zarr group within it. This path will be fed
          into the EMData class.
        :type zarr_path: str

        :param bias: A factor biasing the labelling of vesicle candidates. It labels a candidate as PC- if
          maxima_pos is less than bias * maxima_neg. Otherwise, it labels it as PC+. So, a bias greater than 1 favours
          PC- labelling while a bias less than 1 favours PC+.
        :type bias: int | float | None

        :param save_all_labels_in_one_tiff_file: If True, it saves all predicted labels in one tiff file. The default is
          False.
        :type save_all_labels_in_one_tiff_file: bool

        :param tiff_file_name_of_all_labels: The name of the tiff file.
        :type tiff_file_name_of_all_labels: str | None

        :param save_different_labels_in_different_tiff_files: If True, it saves the different predicted labels in
          different tiff files. One tiff file per label class. The default is False.
        :type save_different_labels_in_different_tiff_files: bool

        :param tiff_file_names_of_different_labels: The names of the tiffs files.
        :type tiff_file_names_of_different_labels: str | list[str] | tuple[str] | None

        :param dtype_labels: Optional numpy data type of the predicted labels. If it is None (Default), numpy will
          decide it (usually int64).
        :type dtype_labels: None | str | np.dtype

        :param show: If True, it shows the predicted label on the raw image by napari. Default is False.
        :type show: bool

        """

        if bias is None:
            bias = 1.0
        elif isinstance(bias, int):
            bias = float(bias)
        elif isinstance(bias, float):
            pass
        else:
            raise TypeError("bias must be an int, a float or None")

        if isinstance(save_all_labels_in_one_tiff_file, bool):

            if save_all_labels_in_one_tiff_file:

                if tiff_file_name_of_all_labels is None:
                    tiff_file_name_of_all_labels = os.path.join(
                        'predicted_labels', 'tiffs', 'bias_{bias:0.3f}'.format(bias=bias), 'labels_all.tif')
                elif not isinstance(tiff_file_name_of_all_labels, str):
                    raise TypeError('tiff_file_name_of_all_labels must be a string or None')
        else:
            raise TypeError('save_all_labels_in_one_tiff_file must be a bool')

        if isinstance(save_different_labels_in_different_tiff_files, bool):
            if save_different_labels_in_different_tiff_files:

                if tiff_file_names_of_different_labels is None:
                    n_tiff_file_names_of_different_labels = None
                elif isinstance(tiff_file_names_of_different_labels, str):
                    tiff_file_names_of_different_labels = [tiff_file_names_of_different_labels]
                    n_tiff_file_names_of_different_labels = 1
                elif isinstance(tiff_file_names_of_different_labels, (list, tuple)):
                    n_tiff_file_names_of_different_labels = len(tiff_file_names_of_different_labels)
                    for i in range(0, n_tiff_file_names_of_different_labels, 1):
                        if not isinstance(tiff_file_names_of_different_labels[i], str):
                            raise TypeError('Each Element of tiff_file_names_of_different_labels must be a string')
                else:
                    raise TypeError('tiff_file_names_of_different_labels must be a None, string, list or tuple')
            else:
                n_tiff_file_names_of_different_labels = None
        else:
            raise TypeError('save_different_labels_in_different_tiff_files must be a bool')

        if not isinstance(show, bool):
            raise TypeError('show must be a bool')

        data = EMData(zarr_path, 'predict', clahe=TRAINING_CONFIG.clahe)
        candidates = None

        # Check if there are multiple channels within the raw data.
        # This shouldn't be the case for us as EM data is 'colourblind'.
        if len(data.raw_data.shape) == 3:
            raw_channels = 1
        elif len(data.raw_data.shape) == 4:
            raw_channels = data.raw_data.shape[0]

        # Get an instance of the model
        detection_model = DetectionModel(
            raw_num_channels = raw_channels,
            voxel_size = data.voxel_size)

        # Initiate a prediction
        predictor = Prediction(
            data = data,
            model = detection_model,
            input_shape = TRAINING_CONFIG.input_shape,
            checkpoint = self.model_checkpoint)

        # Display the border of the output predicition compared to input shape
        #predictor.print_border_message()

        # Get probabilities
        ret = predictor.predict_pipeline()
        probs = torch.nn.Softmax(dim=0)(torch.tensor(ret['prediction'].data)).detach().numpy()
        pos_pred_data = probs[1,:,:,:]
        neg_pred_data = probs[2,:,:,:]

        # Post process with hough detector
        hough_detection = HoughDetector(
            pred_pos = pos_pred_data,
            pred_neg = neg_pred_data,
            voxel_size = data.voxel_size,
            bias = bias)

        hough_detection.process()
        hough_pred = hough_detection.prediction_result

        if dtype_labels is not None:
            hough_pred = hough_pred.astype(dtype=dtype_labels)

        candidates = hough_detection.accepted_candidates

        date = datetime.today().strftime('%Y-%m-%d')

        # Create save location
        save_path = create_unique_directory_file(
            data_path + '/predict/Predictions/bias_{bias:0.3f}_date_{date:s}'.format(bias=bias, date=date))

        save_location = os.path.relpath(save_path, data_path + '/predict')

        # Save the validation prediction in zarr dictionary.
        f = zarr.open(data_path + '/predict', mode='r+')
        f[save_location + '/Hough_transformed'] = hough_pred

        for atr in data.raw_data.attrs:
            f[save_location + '/Hough_transformed'].attrs[atr] = data.raw_data.attrs[atr]

        # Save a single tiff file with all label classes
        if save_all_labels_in_one_tiff_file:
            dirname_tiff = os.path.dirname(tiff_file_name_of_all_labels)
            if len(dirname_tiff) > 0:
                os.makedirs(dirname_tiff, exist_ok=True)
            skimage.io.imsave(tiff_file_name_of_all_labels, hough_pred)

        # Save a tiff file per label class, excluding the background label
        if save_different_labels_in_different_tiff_files:

            n_label_classes = probs.shape[0]
            label_classes = [l for l in range(0, n_label_classes, 1)]

            label_classes_no_bg = [l for l in range(0, n_label_classes, 1) if l != self.label_background]

            n_label_classes_no_bg = len(label_classes_no_bg)

            if tiff_file_names_of_different_labels is None:
                dirname_bias = os.path.join('predicted_labels', 'tiffs', 'bias_{bias:0.3f}'.format(bias=bias))
                tiff_file_names_of_different_labels = [os.path.join(
                    dirname_bias, 'labels_{label:0>3d}.tif'.format(label=label_l))
                    for label_l in label_classes_no_bg]
                n_tiff_file_names_of_different_labels = len(tiff_file_names_of_different_labels)
            elif n_tiff_file_names_of_different_labels == n_label_classes_no_bg:
                pass
            else:
                raise TypeError(
                    'tiff_file_names_of_different_labels must have the same number file names as the label classes '
                    'predicted by the model, excluding the background label.')

            for l in range(0, n_label_classes_no_bg, 1):

                hough_pred_l = np.full(shape=hough_pred.shape, fill_value=self.label_background, dtype=hough_pred.dtype)
                hough_pred_l[hough_pred == label_classes_no_bg[l]] = label_classes_no_bg[l]

                dirname_tiff_l = os.path.dirname(tiff_file_names_of_different_labels[l])
                if len(dirname_tiff_l) > 0:
                    os.makedirs(dirname_tiff_l, exist_ok=True)

                skimage.io.imsave(tiff_file_names_of_different_labels[l], hough_pred_l)

        if show:
            imshow_napari_prediction(zarr_path, save_path)

        return candidates

    def single_image_multi_biases(
            self, zarr_path, biases=1.0,
            save_all_labels_in_one_tiff_file=False, tiff_file_name_of_all_labels=None,
            save_different_labels_in_different_tiff_files=False,
            tiff_file_names_of_different_labels=None,
            dtype_labels=None, show=False):

        # TODO: DOC STRING NEEDS TO UPDATED
        """Use a pretrained vesicle detection model to predict vesicles in unlabelled data by using different biases.

        :param zarr_path: Path to the zarr group that contains the 'predict' zarr group within it. This path will be fed
          into the EMData class.
        :type zarr_path: str

        :param biases: A list of factors biasing the labelling of vesicle candidates. It labels a candidate as PC- if
          maxima_pos is less than biases[b] * maxima_neg. Otherwise, it labels it as PC+. So, biases greater than 1
          favour PC- labelling while biases less than 1 favour PC+.
        :type biases: int | float | list | tuple | None

        :param save_all_labels_in_one_tiff_file: If True, it saves all predicted labels in one tiff file. The default is
          False.
        :type save_all_labels_in_one_tiff_file: bool

        :param tiff_file_name_of_all_labels: The name of the tiff file.
        :type tiff_file_name_of_all_labels: str | None

        :param save_different_labels_in_different_tiff_files: If True, it saves the different predicted labels in
          different tiff files. One tiff file per label class. The default is False.
        :type save_different_labels_in_different_tiff_files: bool

        :param tiff_file_names_of_different_labels: The names of the tiffs files.
        :type tiff_file_names_of_different_labels: str | list[str] | tuple[str] | None

        :param dtype_labels: Optional numpy data type of the predicted labels. If it is None (Default), numpy will
          decide it (usually int64).
        :type dtype_labels: None | str | np.dtype

        :param show: If True, it shows the predicted label on the raw image by napari. Default is False.
        :type show: bool

        """

        if biases is None:
            biases = 1.0
        elif isinstance(biases, int):
            biases = [float(biases)]
        elif isinstance(biases, float):
            biases = [biases]

        elif isinstance(biases, list):
            pass
        elif isinstance(biases, tuple):
            biases = list(biases)
        else:
            raise TypeError('biases must be None, an int, a float, a list or a tuple')

        n_biases = len(biases)

        candidates = [None for b in range(0, n_biases, 1)]  # type: list
        for b in range(0, n_biases, 1):

            candidates[b] = self.single_image_single_bias(
                zarr_path=zarr_path, bias=biases[b],

                save_all_labels_in_one_tiff_file=save_all_labels_in_one_tiff_file,

                tiff_file_name_of_all_labels=(
                    None
                    if (not save_all_labels_in_one_tiff_file) or (tiff_file_name_of_all_labels is None)
                    else tiff_file_name_of_all_labels[b]),

                save_different_labels_in_different_tiff_files=save_different_labels_in_different_tiff_files,

                tiff_file_names_of_different_labels=(
                    None
                    if (not save_different_labels_in_different_tiff_files) or
                       (tiff_file_names_of_different_labels is None)
                    else tiff_file_names_of_different_labels[b]),

                dtype_labels=dtype_labels, show=show)

        return candidates

if __name__ == "__main__":
        
    data_path = input("Provide path to zarr container: ")

    print("-----")

    model_checkpoint = input("Provide the path to the model checkpoint: ")

    print("-----")

    visualise = input("Would you like to visualise the prediction? (y/n): ")

    while visualise.lower() != 'y' and visualise.lower() != 'n':
        print("-----")
        print("Invalid input. Please enter 'y' or 'n' only.")
        visualise = input("Would you like to visualise the prediction? (y/n): ")
    else:
        show = visualise.lower() == 'y'

    biases = [1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4]
    biases = [5]
    print("-----")

    apply = Apply(model_checkpoint=model_checkpoint, label_background=0)

    candidates = apply(
        zarr_path=data_path, biases=biases,
        save_all_labels_in_one_tiff_file=True, tiff_file_name_of_all_labels=None,
        save_different_labels_in_different_tiff_files=True, tiff_file_names_of_different_labels=None,
        dtype_labels='int8', show=show)

    for b in range(0, len(biases), 1):
        pos_labels = 0
        neg_labels = 0
        for candidate in candidates[b]:
            if candidate.label == 1:
                pos_labels += 1 
            if candidate.label == 2:
                neg_labels +=1 

        print('    '.join([
            f"bias: {biases[b]:0.3f}",
            f"PC+ predictions: {pos_labels: >9d}",
            f"PC- predictions: {neg_labels: >9d}"]))
