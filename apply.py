
import argparse
import os
from datetime import datetime
import json
import numpy as np
import skimage.io
# import imageio
import zarr
import torch

from src.data_loader import EMData
from src.processing.predict import Prediction
from src.model.model import DetectionModel
from src.processing.post_processing.hough_detector import HoughDetector
from src.directory_organisor import create_unique_directory_file
from src.visualisation import show_prediction, Viewer

from config.load_configs import TRAINING_CONFIG


class Apply:
    def __init__(self, model_filename, label_background=0, n_label_classes=None):
        """

        :param model_filename: Path to the model that should be used for prediction.
        :type model_filename: str
        :param label_background: The label value for the background pixels/voxels. Default is 0.
        :type label_background: int | None
        """

        self.model_filename = model_filename

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
            dtype_labels=None, do_show=False):

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

        :param do_show: If True, it shows the predicted label on the raw image by napari. Default is False.
        :type do_show: bool

        """

        return None

    def predict_labels(
            self, data, bias=1.0,
            do_save_multi_label_zarr=True, dirname_of_multi_label_zarr=None,
            do_save_multi_label_tiff=False, file_name_of_multi_label_tiff=None,
            do_save_single_label_tiffs=False,
            file_name_of_single_label_tiffs=None,
            dtype_labels=None, do_show=False):

        # TODO: DOC STRING NEEDS TO UPDATED
        """Use a pretrained vesicle detection model to predict vesicles in unlabelled data by using different bias.

        :param data: The raw data to be predicted.
        :type data: EMData

        :param bias: A list of factors biasing the labelling of vesicle candidates. It labels a candidate as PC- if
          maxima_pos is less than bias[b] * maxima_neg. Otherwise, it labels it as PC+. So, bias greater than 1
          favour PC- labelling while bias less than 1 favour PC+.
        :type bias: int | float | list | tuple | None

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

        :param do_show: If True, it shows the predicted label on the raw image by napari. Default is False.
        :type do_show: bool

        """

        # todo: check and format of the arguments

        if bias is None:
            bias = [1.0]
        elif isinstance(bias, int):
            bias = [float(bias)]
        elif isinstance(bias, float):
            bias = [bias]

        elif isinstance(bias, list):
            pass
        elif isinstance(bias, tuple):
            bias = list(bias)
        else:
            raise TypeError('bias must be None, an int, a float, a list or a tuple')

        n_biases = len(bias)
        for b in range(0, n_biases, 1):
            if bias[b] is None:
                bias[b] = 1.0
            elif isinstance(bias[b], int):
                bias[b] = float(bias[b])
            elif isinstance(bias[b], float):
                pass
            else:
                raise TypeError("bias[b] must be an int, a float or None")

        probs = self.predict_probs(data=data)

        data_voxel_size = data.voxel_size
        data_attrs = data.raw_data.attrs
        data_zarr_path = data.zarr_path

        if isinstance(do_show, bool):
            if do_show:
                viewer = Viewer(raw_image=data.numpy(), name='Raw Image', opacity=1.0)
            else:
                viewer = None
        else:
            raise TypeError('do_show must be a bool')

        del data

        labels = [None for b in range(0, n_biases, 1)]  # type: list
        candidates = [None for b in range(0, n_biases, 1)]  # type: list

        for b in range(0, n_biases, 1):

            labels[b], candidates[b] = self.hough_detection(
                probs=probs, voxel_size=data_voxel_size, bias=bias[b], dtype_labels=dtype_labels)

            self.save_labels(
                labels=labels[b],

                do_save_multi_label_zarr=do_save_multi_label_zarr,
                dirname_of_multi_label_zarr=(
                    None if (not do_save_multi_label_zarr)
                    else data_zarr_path if (dirname_of_multi_label_zarr is None)
                    else dirname_of_multi_label_zarr[b]),
                raw_data_attrs=data_attrs,

                do_save_multi_label_tiff=do_save_multi_label_tiff,
                file_name_of_multi_label_tiff=(
                    None if (not do_save_multi_label_tiff) or (file_name_of_multi_label_tiff is None)
                    else file_name_of_multi_label_tiff[b]),

                do_save_single_label_tiffs=do_save_single_label_tiffs,
                file_name_of_single_label_tiffs=(
                    None if (not do_save_single_label_tiffs) or (file_name_of_single_label_tiffs is None)
                    else file_name_of_single_label_tiffs[b]),

                bias=bias[b])

            if do_show:
                viewer.append_labels(labels=labels[b], name=f'labels_with_bias_{bias[b]:0.3f}', opacity=0.4)

        if do_show:
            viewer.show()

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
            checkpoint = self.model_filename)

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
        :type voxel_size: int | float | list[int | float] | tuple[int | float]

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
        elif issubclass(type(voxel_size), (list, tuple)):
            n_sizes = len(voxel_size)
            for s in range(0, n_sizes, 1):
                if isinstance(voxel_size[s], (int, float)):
                    pass
                else:
                    raise TypeError(f"voxel_size[{s:d}] must be an int or float")
        else:
            raise TypeError("voxel_size must be an int, a float or sequence of ints or floats")

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
        elif isinstance(bias, (int, float)):
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
            elif isinstance(bias, (int, float)):
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

        skimage.io.imsave(file_name, labels, check_contrast=False, plugin='tifffile')
        # imageio.volsave(uri=file_name, im=labels, format="tifffile")

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
            elif isinstance(bias, (int, float)):
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

            skimage.io.imsave(file_names[l], labels_l, check_contrast=False, plugin='tifffile')
            # imageio.volsave(uri=file_names[l], im=labels_l, format="tifffile")

        return None

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog=None, usage=None, description=None, epilog=None, parents=[],
        formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None,
        argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True,
        exit_on_error=True)

    parser.add_argument(
        'data_dirname', action='store', type=str, help='The directory path of the zarr data.')

    parser.add_argument(
        'model_filename', action='store', type=str, help='The file path of the trained model.')

    parser.add_argument(
        '-v', '--visualise', action='store_true', type=bool, required=False,
        help='If either "-v" or "--visualise" are in the arguments, visualise the predicted results.')


    parser.add_argument(
        '-b', '--bias', action='store', default=None, type=str, required=False,
        help=(
            'A factor biasing the labelling of vesicle candidates.\n'
            'A vesicle candidate is labelled as PC- if maxima_pos is less than bias * maxima_neg. Otherwise, it is\n'
            'labelled as PC+. So, a bias greater than 1 favours PC- labelling while a bias less than 1 favours PC+.\n'
            'The bias can either be an int, a float or a list of ints and floats in the form:\n'
            '  [bias_1, bias_2, ..., bias_n]\n'
            'For instance, it could be:\n'
            '  [1, 1.5, 2]')
    )


    args = parser.parse_args()

    args.data_dirname
    args.model_filename
    args.visualise

    args.bias = json.loads(args.bias)

    # data_dirname = input("Provide path to zarr container: ")
    # print("-----")
    # model_filename = input("Provide the path to the model checkpoint: ")
    # print("-----")
    # visualise = input("Would you like to visualise the prediction? (y/n): ")
    #
    # while visualise.lower() != 'y' and visualise.lower() != 'n':
    #     print("-----")
    #     print("Invalid input. Please enter 'y' or 'n' only.")
    #     visualise = input("Would you like to visualise the prediction? (y/n): ")
    # else:
    #     do_show = visualise.lower() == 'y'

    print("-----")



    apply = Apply(model_filename=args.model_filename, label_background=0)

    data = EMData(args.data_dirname, 'predict', clahe=TRAINING_CONFIG.clahe)

    probs, labels, candidates = apply(
        data=data, bias=args.bias,
        do_save_multi_label_zarr=True, dirname_of_multi_label_zarr=None,
        do_save_multi_label_tiff=True, file_name_of_multi_label_tiff=None,
        do_save_single_label_tiffs=True, file_name_of_single_label_tiffs=None,
        dtype_labels='int8', do_show=args.visualise)

    for b in range(0, len(args.bias), 1):
        pos_labels = 0
        neg_labels = 0
        for candidate in candidates[b]:
            if candidate.label == 1:
                pos_labels += 1 
            if candidate.label == 2:
                neg_labels +=1 

        print('    '.join([
            f"bias: {args.bias[b]:0.3f}",
            f"PC+ predictions: {pos_labels: >9d}",
            f"PC- predictions: {neg_labels: >9d}"]))


# todo:

# - if main, parse input arguments

# - update the doc stings of the functions

# - format all input arguments of the functions before use

# - define the zarr path outside the function


# done
# - force skimage.io.imsave to save labels as tiff
# - update the visualiser to add multiple layers of labels