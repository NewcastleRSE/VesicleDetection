
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
from src.visualisation import show_prediction

from config.load_configs import TRAINING_CONFIG


class Apply:
    def __init__(self, model_checkpoint, label_background=0, n_label_classes=None):
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

        self._are_label_classes_not_initiated = True

        self.n_label_classes = None
        self.label_classes = None
        self.label_classes_no_bg = None
        self.n_label_classes_no_bg = None

        self.init_label_classes(n_label_classes=n_label_classes)

    def init_label_classes(self, n_label_classes):

        if self._are_label_classes_not_initiated:

            if n_label_classes is None:
                pass
            elif isinstance(n_label_classes, int):
                self.n_label_classes = n_label_classes
                self.label_classes = [l for l in range(0, self.n_label_classes, 1)]

                self.label_classes_no_bg = [l for l in range(0, self.n_label_classes, 1) if l != self.label_background]

                self.n_label_classes_no_bg = len(self.label_classes_no_bg)

                self._are_label_classes_not_initiated = False
            else:
                raise TypeError('n_label_classes must be None or an int')

        return None

    def __call__(self, *args, **kwargs):

        return self.predict_labels(*args, **kwargs)

    def single_image_single_bias(
            self, data, bias=1.0,
            save_all_labels_in_one_tiff_file=False, tiff_file_name_of_all_labels=None,
            save_different_labels_in_different_tiff_files=False,
            tiff_file_names_of_different_labels= None,
            dtype_labels=None, show=False):

        """Use a pretrained vesicle detection model to predict vesicles in unlabelled data by using a single bias.

        :param data: The raw data to be predicted.
        :type data: EMData

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

        probs = self.predict_probs(data=data)

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
            skimage.io.imsave(tiff_file_name_of_all_labels, hough_pred, plugin='tifffile', check_contrast=False)

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

                skimage.io.imsave(
                    tiff_file_names_of_different_labels[l], hough_pred_l, plugin='tifffile', check_contrast=False)

        if show:
            show_prediction(data.numpy(), hough_pred)

        return candidates

    def predict_labels(
            self, data, biases=1.0,
            do_save_multi_label_zarr=True, dirname_of_multi_label_zarr=None,
            do_save_multi_label_tiff=False, file_name_of_multi_label_tiff=None,
            do_save_single_label_tiffs=False,
            file_name_of_single_label_tiffs=None,
            dtype_labels=None, show=False):

        # TODO: DOC STRING NEEDS TO UPDATED
        """Use a pretrained vesicle detection model to predict vesicles in unlabelled data by using different biases.

        :param data: The raw data to be predicted.
        :type data: EMData

        :param biases: A list of factors biasing the labelling of vesicle candidates. It labels a candidate as PC- if
          maxima_pos is less than biases[b] * maxima_neg. Otherwise, it labels it as PC+. So, biases greater than 1
          favour PC- labelling while biases less than 1 favour PC+.
        :type biases: int | float | list | tuple | None

        :param do_save_multi_label_tiff: If True, it saves all predicted labels in one tiff file. The default
          is False.
        :type do_save_multi_label_tiff: bool

        :param file_name_of_multi_label_tiff: The name of the tiff file.
        :type file_name_of_multi_label_tiff: list[str] | tuple[str] | None

        :param do_save_single_label_tiffs: If True, it saves the different predicted labels in
          different tiff files. One tiff file per label class. The default is False.
        :type do_save_single_label_tiffs: bool

        :param file_name_of_single_label_tiffs: The names of the tiffs files.
        :type file_name_of_single_label_tiffs: list[list[str]] | tuple[tuple[str]] | None

        :param dtype_labels: Optional numpy data type of the predicted labels. If it is None (Default), numpy will
          decide it (usually int64).
        :type dtype_labels: None | str | np.dtype

        :param show: If True, it shows the predicted label on the raw image by napari. Default is False.
        :type show: bool

        """

        # todo: check and format of the arguments

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
        for b in range(0, n_biases, 1):
            if biases[b] is None:
                biases[b] = 1.0
            elif isinstance(biases[b], int):
                biases[b] = float(biases[b])
            elif isinstance(biases[b], float):
                pass
            else:
                raise TypeError("biases[b] must be an int, a float or None")

        probs = self.predict_probs(data=data)

        labels = [None for b in range(0, n_biases, 1)]  # type: list
        candidates = [None for b in range(0, n_biases, 1)]  # type: list

        for b in range(0, n_biases, 1):

            labels[b], candidates[b] = self.hough_detection(
                probs=probs, voxel_size=data.voxel_size, bias=biases[b], dtype_labels=dtype_labels)

            self.save_labels(
                labels=labels[b],

                do_save_multi_label_zarr=do_save_multi_label_zarr,
                dirname_of_multi_label_zarr=(
                    None if (not do_save_multi_label_zarr) or (dirname_of_multi_label_zarr is None)
                    else dirname_of_multi_label_zarr[b]),

                raw_data_attrs=data.raw_data.attrs,

                do_save_multi_label_tiff=do_save_multi_label_tiff,
                file_name_of_multi_label_tiff=(
                    None if (not do_save_multi_label_tiff) or (file_name_of_multi_label_tiff is None)
                    else file_name_of_multi_label_tiff[b]),

                do_save_single_label_tiffs=do_save_single_label_tiffs,
                file_name_of_single_label_tiffs=(
                    None if (not do_save_single_label_tiffs) or (file_name_of_single_label_tiffs is None)
                    else file_name_of_single_label_tiffs[b]),

                bias=biases[b])

        if show:
            show_prediction(data.numpy(), labels)

        return probs, labels, candidates

    def predict_probs(self, data):
        """Use a pretrained vesicle detection model to predict vesicles in unlabelled data by using different biases.

        :param data: The raw data to be predicted.
        :type data: EMData

        """

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

        if self._are_label_classes_not_initiated:
            self.init_label_classes(n_label_classes=probs.shape[0])

        return probs

    def hough_detection(self, probs, voxel_size, bias=1.0, dtype_labels=None):

        """Use a pretrained vesicle detection model to predict vesicles in unlabelled data by using a single bias.

        :param probs: An Array containing the model-predicted probability of the labels.
        :type probs: np.ndarray | torch.Tensor

        :param voxel_size: The voxel size of the image.
        :type voxel_size: int | float

        :param bias: A factor biasing the labelling of vesicle candidates. It labels a candidate as PC- if
          maxima_pos is less than bias * maxima_neg. Otherwise, it labels it as PC+. So, a bias greater than 1 favours
          PC- labelling while a bias less than 1 favours PC+.
        :type bias: int | float | None

        :param dtype_labels: Optional numpy data type of the predicted labels. If it is None (Default), numpy will
          decide it (usually int64).
        :type dtype_labels: None | str | np.dtype

        """

        if isinstance(probs, np.ndarray):
            pass
        elif isinstance(probs, torch.Tensor):
            probs = probs.detach().numpy()
        else:
            raise TypeError("probs must be a numpy array or torch.Tensor")

        if isinstance(voxel_size, (int, float)):
            pass
        else:
            raise TypeError("voxel_size must be a float or int")

        if bias is None:
            bias = 1.0
        elif isinstance(bias, int):
            bias = float(bias)
        elif isinstance(bias, float):
            pass
        else:
            raise TypeError("bias must be an int, a float or None")

        pos_pred_data = probs[1, :, :, :]
        neg_pred_data = probs[2, :, :, :]

        # Post process with hough detector
        hough_detection = HoughDetector(
            pred_pos=pos_pred_data,
            pred_neg=neg_pred_data,
            voxel_size=voxel_size,
            bias=bias)

        hough_detection.process()
        hough_pred = hough_detection.prediction_result

        if dtype_labels is not None:
            hough_pred = hough_pred.astype(dtype=dtype_labels)

        # candidates = None
        candidates = hough_detection.accepted_candidates

        return hough_pred, candidates

    def save_labels(
            self, labels,
            do_save_multi_label_zarr=True, dirname_of_multi_label_zarr=None, raw_data_attrs=None,
            do_save_multi_label_tiff=False, file_name_of_multi_label_tiff=None,
            do_save_single_label_tiffs=False, file_name_of_single_label_tiffs=None,
            bias=None):

        """Use a pretrained vesicle detection model to predict vesicles in unlabelled data by using a single bias.

        :param labels: An Array containing the labels of the raw image.
        :type labels: np.ndarray | torch.Tensor

        :param do_save_multi_label_tiff: If True, it saves all predicted labels in one tiff file. The default is
          False.
        :type do_save_multi_label_tiff: bool

        :param file_name_of_multi_label_tiff: The name of the tiff file.
        :type file_name_of_multi_label_tiff: str | None

        :param do_save_single_label_tiffs: If True, it saves the different predicted labels in
          different tiff files. One tiff file per label class. The default is False.
        :type do_save_single_label_tiffs: bool

        :param file_name_of_single_label_tiffs: The names of the tiffs files.
        :type file_name_of_single_label_tiffs: str | list[str] | tuple[str] | None

        :param bias: The bias used in the hough detection.
        :type bias: int | float | None

        """

        if do_save_multi_label_zarr:
            self.save_multi_label_zarr(
                labels=labels, dirname_zarr=dirname_of_multi_label_zarr, raw_data_attrs=raw_data_attrs, bias=bias)

        if isinstance(do_save_multi_label_tiff, bool):
            if do_save_multi_label_tiff:
                self.save_multi_label_tiff(labels=labels, file_name=file_name_of_multi_label_tiff, bias=bias)
        else:
            raise TypeError('do_save_multi_label_tiff must be a bool')

        if isinstance(do_save_single_label_tiffs, bool):
            if do_save_single_label_tiffs:
                self.save_single_label_tiffs(labels=labels, file_names=file_name_of_single_label_tiffs, bias=bias)
        else:
            raise TypeError('do_save_single_label_tiffs must be a bool')

        return None

    def save_multi_label_zarr(self, labels, dirname_zarr, raw_data_attrs, bias=None):

        """Use a pretrained vesicle detection model to predict vesicles in unlabelled data by using a single bias.

        :param labels: An Array containing the labels of the raw image.
        :type labels: np.ndarray | torch.Tensor

        :param dirname_zarr: Path to the zarr group that contains the 'predict' zarr group within it. This path will be fed
          into the EMData class.
        :type dirname_zarr: str

        :param raw_data_attrs: The attributes of the zarr container. They are stored in data.raw_data.attrs.

        :param bias: The bias used in the hough detection.
        :type bias: int | float | None
        """

        if isinstance(labels, np.ndarray):
            pass
        elif isinstance(labels, torch.Tensor):
            labels = labels.detach().numpy()
        else:
            raise TypeError("labels must be a numpy array or torch.Tensor")

        date = datetime.today().strftime('%Y-%m-%d')

        # Create save location

        save_path = os.path.join(dirname_zarr, 'predict', 'Predictions')

        if bias is None:
            save_path = os.path.join(save_path, 'date_{date:s}'.format(date=date))
        elif isinstance(bias, int):
            save_path = os.path.join(save_path, 'bias_{bias:0.3f}_date_{date:s}'.format(bias=bias, date=date))
        else:
            raise TypeError('bias must be an int or None')

        save_path = create_unique_directory_file(save_path)

        save_location = os.path.relpath(save_path, dirname_zarr + '/predict')

        # Save the validation prediction in zarr dictionary.
        f = zarr.open(dirname_zarr + '/predict', mode='r+')
        f[save_location + '/Hough_transformed'] = labels

        for atr in raw_data_attrs:
            f[save_location + '/Hough_transformed'].attrs[atr] = raw_data_attrs[atr]

        return None

    def save_multi_label_tiff(self, labels, file_name=None, bias=None):

        """Use a pretrained vesicle detection model to predict vesicles in unlabelled data by using a single bias.

        :param labels: An Array containing the labels of the raw image.
        :type labels: np.ndarray | torch.Tensor

        :param file_name: The names of the files.
        :type file_name: str | None

        :param bias: The bias used in the hough detection.
        :type bias: int | float | None
        """

        if isinstance(labels, np.ndarray):
            pass
        elif isinstance(labels, torch.Tensor):
            labels = labels.detach().numpy()
        else:
            raise TypeError("labels must be a numpy array or torch.Tensor")

        if file_name is None:
            dirname = os.path.join('predicted_labels', 'tiffs')
            if bias is None:
                pass
            elif isinstance(bias, int):
                dirname = os.path.join(dirname, 'bias_{bias:0.3f}'.format(bias=bias))
            else:
                raise TypeError('bias must be an int or None')

            file_name = os.path.join(dirname, 'labels_all.tif')

        elif isinstance(file_name, str):
            dirname = os.path.dirname(file_name)
        else:
            raise TypeError('file_name must be a string or None')

        if len(dirname) > 0:
            os.makedirs(dirname, exist_ok=True)

        skimage.io.imsave(file_name, labels, plugin='tifffile', check_contrast=False)

        return None

    def save_single_label_tiffs(self, labels, file_names=None, bias=None):

        """Use a pretrained vesicle detection model to predict vesicles in unlabelled data by using a single bias.

        :param labels: An Array containing the labels of the raw image.
        :type labels: np.ndarray | torch.Tensor

        :param file_names: The names of the files.
        :type file_names: str | list[str] | tuple[str] | None

        :param bias: The bias used in the hough detection.
        :type bias: int | float | None
        """

        if isinstance(labels, np.ndarray):
            pass
        elif isinstance(labels, torch.Tensor):
            labels = labels.detach().numpy()
        else:
            raise TypeError("labels must be a numpy array or torch.Tensor")

        if file_names is None:
            n_file_names = None
        elif isinstance(file_names, str):
            file_names = [file_names]
            n_file_names = 1
        elif isinstance(file_names, (list, tuple)):
            n_file_names = len(file_names)
            for i in range(0, n_file_names, 1):
                if not isinstance(file_names[i], str):
                    raise TypeError('Each Element of file_names must be a string')
        else:
            raise TypeError('file_names must be a None, string, list or tuple')


        # Save a tiff file per label class, excluding the background label

        if file_names is None:

            dirname = os.path.join('predicted_labels', 'tiffs')
            if bias is None:
                pass
            elif isinstance(bias, int):
                dirname = os.path.join(dirname, 'bias_{bias:0.3f}'.format(bias=bias))
            else:
                raise TypeError('bias must be an int or None')

            file_names = [
                os.path.join(dirname, 'labels_{label:0>3d}.tif'.format(label=label_l))
                for label_l in self.label_classes_no_bg
            ]
            n_file_names = len(file_names)
        elif n_file_names == self.n_label_classes_no_bg:
            pass
        else:
            raise TypeError(
                'file_names must have the same number file names as the label classes '
                'predicted by the model, excluding the background label.')

        for l in range(0, self.n_label_classes_no_bg, 1):

            labels_l = np.full(shape=labels.shape, fill_value=self.label_background, dtype=labels.dtype)
            labels_l[labels == self.label_classes_no_bg[l]] = self.label_classes_no_bg[l]

            dirname_tiff_l = os.path.dirname(file_names[l])
            if len(dirname_tiff_l) > 0:
                os.makedirs(dirname_tiff_l, exist_ok=True)

            skimage.io.imsave(file_names[l], labels_l, plugin='tifffile', check_contrast=False)

        return None

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

    biases = [5]

    print("-----")

    apply = Apply(model_checkpoint=model_checkpoint, label_background=0)

    data = EMData(data_path, 'predict', clahe=TRAINING_CONFIG.clahe)

    candidates = apply(
        data=data, biases=biases,
        do_save_multi_label_tiff=True, file_name_of_multi_label_tiff=None,
        do_save_single_label_tiffs=True, file_name_of_single_label_tiffs=None,
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


# todo:
# - update the visualiser to add multiple layers of labels
# - define the zarr path outside the function
# - update the doc stings of the functions
# - format all input argument of the functions before use

# done
# - force skimage.io.imsave to save labels as tiff